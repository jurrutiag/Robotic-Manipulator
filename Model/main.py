

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
    options = ['Run all, testing', 'Run All', 'Initialize only', 'Profiling']
    option_string = input("Select one: Run All (1), Initialize Only (2), Profiling (3), Testing (Else): ")
    try:
        option = options[int(option_string)]
    except ValueError:
        option = options[0]

    # Cores for multiprocessing
    cores = 1

    # Manipulator parameters
    manipulator_dimensions = [5, 5, 5, 5]
    manipulator_mass = [1, 1, 1]

    # Parameters to change on the json
    parameters_variations = {
        "torques_error_ponderation": [0.0003],
        "pop_size": [100],
        "elitism_size": [10],
        "generation_threshold": [1000],
        "selection_method": ["pareto_tournament"],
        "niche_sigma": [0.8],
        "pareto_tournament_size": [3],
        "cores": [4],
        "generation_for_print": [10],
        "mut_individual_prob": [0.5],
        "sampling_points": [10],
        "desired_position": [[-5, -5, 7]],
        "total_time": [5]
    }

    model_repetition = 1

    ## Execution

    manipulator = RoboticManipulator(manipulator_dimensions, manipulator_mass)

    # Run all
    if option in options[:2]:
        run_name = input("Enter the name for saving the run: ") if option == options[1] else 'json_test'
        save_load_json = JSONSaveLoad(save_filename=run_name,
                                      parameters_variations=parameters_variations)

        save_load_json.loadParameters(model_repetition)

        runs = save_load_json.getRuns()
        print(f"Executing models on {cores} cores...")
        executer = MultiCoreExecuter(runs, manipulator, save_load_json, cores=cores)
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