

if __name__ == "__main__":
    from GeneticAlgorithm import GeneticAlgorithm
    from RoboticManipulator import RoboticManipulator
    import numpy as np
    import matplotlib.pyplot as plt
    from JSONSaveLoad import JSONSaveLoad
    from MultiCoreExecuter import MultiCoreExecuter
    import cProfile

    ## Configurations

    # np.random.seed(0) # for testing

    # Select a running option.
    options = ['Run all, testing', 'Run All', 'Initialize only', 'Profiling', 'Render', 'Find Pareto Frontier']
    option_string = input("Select one: Run All (1), Initialize Only (2), Profiling (3), Render (4), Find Pareto Frontier (5), Testing (Else): ")
    try:
        option = options[int(option_string)]
    except ValueError:
        option = options[0]

    # Cores for multiprocessing
    cores = 1

    # Show info on dedicated screen
    dedicated_screen = True

    # Manipulator parameters
    manipulator_dimensions = [5, 5, 5, 5]
    manipulator_mass = [1, 1, 1]

    # Parameters to change on the json
    parameters_variations = {
        "desired_position": [[5, 5, 5]],
        "cores": [4],
        "pop_size": [100],
        "cross_individual_prob": [0.8],
        "mut_individual_prob": [0.5],
        "cross_joint_prob": [0.25],
        "mut_joint_prob": [0.25],
        "pairing_prob": [0.5],
        "sampling_points": [10],
        "torques_ponderations": [[1, 1, 1, 1]],
        "generation_threshold": [200],
        "fitness_threshold": [1],
        "progress_threshold": [1],
        "generations_progress_threshold": [50],
        "distance_error_ponderation": [1],
        "torques_error_ponderation": [0.1],
        "velocity_error_ponderation": [0],
        "rate_of_selection": [0.3],
        "elitism_size": [5],
        "selection_method": [[0, 0, 0.7, 0.3]],
        "rank_probability": [0.5],
        "pareto_tournament_size": [3],
        "niche_sigma": [0.5],
        "generation_for_print": [10],
        "exponential_initialization": [False],
        "total_time": [5],
        "individuals_to_display": [5]
    }

    model_repetition = 1

    ## Execution

    manipulator = RoboticManipulator(manipulator_dimensions, manipulator_mass)

    # Run all
    if option in options[:2]:
        run_name = input("Enter the name for saving the run: ") if option == options[1] else 'json_test'
        save_load_json = JSONSaveLoad(GA=GeneticAlgorithm(manipulator, [5, 5, 5]),
                                      save_filename=run_name,
                                      parameters_variations=parameters_variations)

        save_load_json.loadParameters(model_repetition)

        runs = save_load_json.getRuns()
        print(f"Executing models on {cores} cores...")
        executer = MultiCoreExecuter(runs, manipulator, save_load_json, cores=cores, dedicated_screen=dedicated_screen)
        executer.run()

    # Initialize only
    elif option == options[2]:
        default_parameters = JSONSaveLoad.loadDefaults()
        GA = GeneticAlgorithm(manipulator, **default_parameters)
        GA.initialization()
        individual = GA.getPopulation()[0]

        for i, ang in enumerate(np.transpose(individual.getGenes())):
            plt.plot(ang)

        plt.legend([r"$\theta_1$", r"$\theta_2$", r"$\theta_3$", r"$\theta_4$"])
        plt.xlabel("Unidad de tiempo")
        plt.ylabel("Radianes")

        plt.show()

    # Profiling
    elif option == options[3]:
        print("Run the following commands:")
        print("python -m cProfile -o main_profiling.cprof main.py")
        print("pyprof2calltree -k -i main_profiling.cprof")

        default_parameters = JSONSaveLoad.loadDefaults()
        default_params_profiling = default_parameters
        default_params_profiling['generation_threshold'] = 1
        default_params_profiling['torques_error_ponderation'] = 0.0003

        GA = GeneticAlgorithm(manipulator, **default_parameters)

        cProfile.run('GA.runAlgorithm(print=False)', sort='cumtime')

    # Rendering
    elif option == options[4]:
        import sys
        sys.path.insert(1, '../Blender')
        from RenderBlender import render
        render()

    # Find Pareto Frontier
    elif option == options[5]:
        import pygmo as pg
        import json

        model = input("Enter model name to find the Pareto frontier: ")
        with open(f"../Model/Trained Models/{model}/{model}.json") as f:
            the_json = json.load(f)

        multi_fitnesses = [(ind["ID"], ind["Multi Fitness"]) for ind in the_json["Best Individuals"]]
        dominants = []

        for fit_id, multi_fitness in multi_fitnesses:
            if not np.any([pg.pareto_dominance(others_fitness, multi_fitness)
                  for _, others_fitness in multi_fitnesses if others_fitness != multi_fitness]):
                dominants.append(fit_id)

        print("Dominant ids: ", *dominants)

    import winsound
    frequency = 2500  # Set Frequency To 2500 Hertz
    duration = 1000  # Set Duration To 1000 ms == 1 second
    winsound.Beep(frequency, duration)