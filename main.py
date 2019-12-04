from Model.GeneticAlgorithm import GeneticAlgorithm
from Model.RoboticManipulator import RoboticManipulator
import numpy as np
import matplotlib.pyplot as plt
from Model.JSONSaveLoad import JSONSaveLoad
from Model.MultiCoreExecuter import MultiCoreExecuter
import json
import sys
import os
from definitions import getModelDir, getTuningDict


def runAll(manipulator, run_name, parameters_variations, processes, run_on_command_line, all_combinations=False, continue_tuning=False, repetitions=1):
    save_load_json = JSONSaveLoad(GA=GeneticAlgorithm(manipulator),
                                  save_filename=run_name,
                                  parameters_variations=parameters_variations)

    save_load_json.loadParameters(all_combinations=all_combinations, continue_tuning=continue_tuning, repetitions=repetitions)

    runs = save_load_json.getRuns()
    print(f"Executing models on {processes} cores...")
    executer = MultiCoreExecuter(runs,
                                 manipulator,
                                 save_load_json,
                                 processes=processes,
                                 dedicated_screen=not run_on_command_line,
                                 model_name=run_name)
    executer.run()



def initializeOnly(manipulator):
    GA = GeneticAlgorithm(manipulator)
    GA.initialization()
    individual = GA.getPopulation()[0]

    fig_init, ax_init = plt.subplots(ncols=1, nrows=1)
    for i, ang in enumerate(np.transpose(individual.getGenes())):
        ax_init.plot(ang)

    ax_init.legend([r"$\theta_1$", r"$\theta_2$", r"$\theta_3$", r"$\theta_4$"])
    ax_init.set_xlabel("Unidad de tiempo")
    ax_init.set_ylabel("Radianes")

    return fig_init

def findDominantsFromModel(model):
    with open(getModelDir(model)) as f:
        the_json = json.load(f)

    multi_fitnesses = [(ind["ID"], ind["Multi Fitness"]) for ind in the_json["Best Individuals"]]
    dominants = []

    for fit_id, multi_fitness in multi_fitnesses:
        if not np.any([pg.pareto_dominance(others_fitness, multi_fitness)
                       for _, others_fitness in multi_fitnesses if others_fitness != multi_fitness]):
            dominants.append(fit_id)

    return dominants


def findDominantsFromTuning(run_name):

    with open(getModelDir(run_name)) as f:
        model_json = json.load(f)
        inds = model_json['Best Individuals']
        first_ind_info = inds[0]['Info']

        final_info_dictionary = {}

        for ind in inds:
            for key, val in ind['Info'].items():
                if val != first_ind_info[key]:
                    if key not in final_info_dictionary:
                        final_info_dictionary[key] = {}

                    key = str(key)
                    val = str(val)
                    if key in final_info_dictionary and val in final_info_dictionary[key]:
                        final_info_dictionary[key][val].append(ind['Multi Fitness'])
                    elif key in final_info_dictionary:
                        final_info_dictionary[key][val] = [ind['Multi Fitness']]
                    else:
                        final_info_dictionary[key] = {val: [ind['Multi Fitness']]}
                    break
            else:
                initial = [ind['Multi Fitness']]

        for key, val in final_info_dictionary.items():
            default_val = first_ind_info[key]
            final_info_dictionary[key][str(default_val)] = initial

            for keyNum, valNum in final_info_dictionary[key].items():
                final_info_dictionary[key][keyNum] = np.mean(valNum, axis=0).tolist()

    import pygmo as pg
    best_params = {}
    for key, val in final_info_dictionary.items():
        best_params[key] = []
        for param_val, multi_fitness in val.items():
            if not np.any([pg.pareto_dominance(others_fitness, multi_fitness)
                           for _, others_fitness in val.items() if others_fitness != multi_fitness]):
                best_params[key].append((param_val, 1 / (
                            1 + np.array(multi_fitness) @ np.array([1, 0.1 * (1 / 256.79), 0 * (1 / 0.39)]))))

    with open(getTuningDict(run_name), 'w') as f:
        json.dump({"all": final_info_dictionary, "best": best_params}, f)


def main(on_display):
    run_on_command_line = not on_display

    ## Configurations

    np.random.seed(0) # for testing

    # Select a running option.
    options = ['Run all, testing', 'Run All', 'Initialize only', 'Profiling', 'Render', 'Find Pareto Frontier',
               'Tune Parámeters']

    # Parameters to change on the json
    parameters_variations = {
        "cores": [8],
        "cross_individual_prob": [0.5],
        "cross_joint_prob": [0.75],
        "desired_position": [[5, 5, 5]],
        "distance_error_ponderation": [1],
        "elitism_size": [15],
        "exponential_initialization": [False],
        "fitness_threshold": [1],
        "generation_for_print": [10],
        "generation_threshold": [2000],
        "generations_progress_threshold": [50],
        "individuals_to_display": [5],
        "mut_individual_prob": [0.5],
        "mut_joint_prob": [0.25],
        "niche_sigma": [0.2],
        "pairing_prob": [0.5],
        "pareto_tournament_size": [5],
        "pop_size": [150],
        "print_data": [True],
        "progress_threshold": [1],
        "rank_probability": [0.4],
        "rate_of_selection": [0.4],
        "sampling_points": [20, 30, 50],
        "selection_method": [[0, 0.2, 0.6, 0.2]],
        "torques_error_ponderation": [0.1],
        "torques_ponderations": [[1, 1, 1, 1]],
        "total_time": [5],
        "velocity_error_ponderation": [0]
    }

    all_combinations = False
    continue_tuning = False
    repetitions = 1
    processes = 1

    # Manipulator parameters

    manipulator_dimensions = [5, 5, 5, 5]
    manipulator_mass = [1, 1, 1]
    manipulator = RoboticManipulator(manipulator_dimensions, manipulator_mass)

    ## Execution

    if run_on_command_line:
        option_string = input(
            "Select one: Run All (1), Initialize Only (2), Profiling (3), Render (4), Find Pareto Frontier (5), Tune Parámeters (6), Testing (Else): ")
        try:
            option = options[int(option_string)]
        except ValueError:
            option = options[0]

        if option in options[:2]:
            run_name = input("Enter the name for saving the run: ") if option == options[1] else 'json_test'
            all_combinations = input("All Combinations? (Y/N)") == "Y"
            continue_tuning = input("Continue tuning? (Y/N)") == "Y"
            try:
                repetitions = int(input("How many times per model? (integer)"))
            except ValueError:
                repetitions = 1

    else:
        sys.path.insert(1, '../InfoDisplay')
        from InfoDisplay.InformationWindow import runMainWindow
        main_window_info = runMainWindow(GeneticAlgorithm(None).getAlgorithmInfo())
        if main_window_info == {}:
            sys.exit(1)

        option_num = main_window_info['final_option']

        if option_num == 1:
            parameters_variations = main_window_info['parameters_variations']
            processes = main_window_info['cores']
            run_name = main_window_info['run_name']
            all_combinations = main_window_info['all_combinations']
            continue_tuning = main_window_info['continue_tuning']
            repetitions = main_window_info['repetitions']
        elif option_num == 4:
            render_model_name = main_window_info['render_model_name']
            render_run = main_window_info['render_run'] if not main_window_info['all_runs'] else -1
            render_individuals = main_window_info['render_individuals']
        elif option_num == 6:
            return

        option = options[option_num]

    # Run all
    if option in options[:2]:
        runAll(manipulator, run_name, parameters_variations, processes, run_on_command_line, all_combinations, continue_tuning,
               repetitions)

    # Initialize only
    elif option == options[2]:
        initializeOnly(manipulator)

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

        sys.path.insert(1, '../Blender')
        from Blender.RenderBlender import render

        if run_on_command_line:
            render_model_name = input("Enter the model name: ")
            render_run = int(input(
                "Enter Run Number: "))
            render_individuals = [int(input(
                "Enter individual to render (integer starting from 0): "))]

        render(render_model_name, render_run, render_individuals)

    # Find Pareto Frontier
    elif option == options[5]:
        import pygmo as pg

        model = input("Enter model name to find the Pareto frontier: ")
        dominants = findDominantsFromModel(model)

        print("Dominant ids: ", *dominants)

    elif option == options[6]:
        run_name = input("Enter Run name: ")

        findDominantsFromTuning(run_name)





if __name__ == "__main__":
    try:
        on_display = sys.argv[1] == "-d"
    except IndexError:
        on_display = False

    main(on_display)