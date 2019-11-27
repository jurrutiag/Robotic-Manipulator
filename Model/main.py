

if __name__ == "__main__":
    from GeneticAlgorithm import GeneticAlgorithm
    from RoboticManipulator import RoboticManipulator
    import numpy as np
    import matplotlib.pyplot as plt
    from JSONSaveLoad import JSONSaveLoad
    from MultiCoreExecuter import MultiCoreExecuter
    import json

    ## Configurations

    # np.random.seed(0) # for testing

    # Select a running option.
    options = ['Run all, testing', 'Run All', 'Initialize only', 'Profiling', 'Render', 'Find Pareto Frontier']

    # Cores for multiprocessing
    cores = 1

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
        "generation_threshold": [100],
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

    run_on_command_line = input("Run on command line? (Y/N)") == "Y"

    if run_on_command_line:
        option_string = input(
            "Select one: Run All (1), Initialize Only (2), Profiling (3), Render (4), Find Pareto Frontier (5), Testing (Else): ")
        try:
            option = options[int(option_string)]
        except ValueError:
            option = options[0]

        if option in options[:2]:
            run_name = input("Enter the name for saving the run: ") if option == options[1] else 'json_test'
            all_combinations = input("All combinations of parameters? (Y/N)") == "Y"
    else:
        import sys
        sys.path.insert(1, '../InfoDisplay')
        from InformationWindow import runMainWindow
        main_window_info = runMainWindow(GeneticAlgorithm(None).getAlgorithmInfo())
        option_num = main_window_info['final_option']

        if option_num is None:
            sys.exit(1)
        elif option_num == 1:
            parameters_variations = main_window_info['parameters_variations']
            cores = main_window_info['cores']
            run_name = main_window_info['run_name']
            all_combinations = main_window_info['all_combinations']
        elif option_num == 4:
            render_model_name = main_window_info['render_model_name']
            render_run = main_window_info['render_run']
            render_individuals = main_window_info['render_individuals']

        option = options[option_num]

    # Run all
    if option in options[:2]:
        save_load_json = JSONSaveLoad(GA=GeneticAlgorithm(manipulator),
                                      save_filename=run_name,
                                      parameters_variations=parameters_variations)

        save_load_json.loadParameters(model_repetition, all_combinations=all_combinations)

        runs = save_load_json.getRuns()
        print(f"Executing models on {cores} cores...")
        executer = MultiCoreExecuter(runs,
                                     manipulator,
                                     save_load_json,
                                     cores=cores,
                                     dedicated_screen=not run_on_command_line,
                                     model_name=run_name)
        executer.run()

        if not all_combinations:
            with open(f'../Model/Trained Models/{run_name}/{run_name}.json') as f:
                model_json = json.load(f)
                inds = model_json['Best Individuals']
                first_ind_info = inds[0]['Info']

                final_info_dictionary = {}

                for ind in inds:
                    for key, val in ind['Info'].items():
                        if val != first_ind_info[key]:
                            if f'{key} = {val}' in final_info_dictionary:
                                final_info_dictionary[f'{key} = {val}'].append(ind['Multi Fitness'])
                            else:
                                final_info_dictionary[f'{key} = {val}'] = [ind['Multi Fitness']]
                            break
                    else:
                        if 'Initial' in final_info_dictionary:
                            final_info_dictionary['Initial'].append(ind['Multi Fitness'])
                        else:
                            final_info_dictionary['Initial'] = [ind['Multi Fitness']]

                for key, val in final_info_dictionary.items():
                    final_info_dictionary[key] = np.mean(val, axis=0).tolist()

                with open(f'../Model/Trained Models/{run_name}/{run_name}_mfitnesses_dict.json', 'w') as f:
                    json.dump(final_info_dictionary, f)

    # Initialize only
    elif option == options[2]:
        GA = GeneticAlgorithm(manipulator)
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
        import cProfile
        print("Run the following commands to see the profiling on the interface:")
        print("python -m cProfile -o main_profiling.cprof main.py")
        print("pyprof2calltree -k -i main_profiling.cprof")

        GA = GeneticAlgorithm(manipulator, print_data=False)

        print("Profiling...")
        cProfile.run('GA.runAlgorithm()', sort='cumtime')

    # Rendering
    elif option == options[4]:
        import sys

        sys.path.insert(1, '../Blender')
        from RenderBlender import render

        if run_on_command_line:
            render_model_name = input("Enter the model name: ")
            render_run = int(input(
                "Enter Run Number: "))
            render_individuals = [int(input(
                "Enter individual to render (integer): "))]

        print(render_model_name, render_run, render_individuals)
        render(render_model_name, render_run, render_individuals)

    # Find Pareto Frontier
    elif option == options[5]:
        import pygmo as pg

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
