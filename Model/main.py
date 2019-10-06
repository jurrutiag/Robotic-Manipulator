

if __name__ == "__main__":
    from GeneticAlgorithm import GeneticAlgorithm
    from RoboticManipulator import RoboticManipulator
    import numpy as np
    import pickle
    import matplotlib.pyplot as plt
    from JSONSaveLoad import JSONSaveLoad
    from MultiCoreExecuter import MultiCoreExecuter

    ## Configurations

    # np.random.seed(0) # for testing

    # True if you want to run all the algorithm, otherwise it will only initialize and show the first individual.
    all = True

    # True if you want to load the parameters from a JSON file
    from_file = True

    # Parameters to change on the json
    # parameters_variations = {
    #     "torques_error_ponderation": [0],
    #     "generation_threshold": [30],
    #     "pop_size": [100, 150],
    #     "cross_individual_prob": [0.5, 0.6, 0.7, 0.8],
    #     "mut_individual_prob": [0.01, 0.05],
    #     "cross_joint_prob": [0.25, 0.5],
    #     "mut_joint_prob": [0.25, 0.5],
    #     "sampling_points": [20]
    # }

    parameters_variations = {
        "torques_error_ponderation": [0],
        "generation_threshold": [30, 40, 50, 60],
        "cores": [1]
    }

    # Name of the set of parameters to run
    run_name = "json_test"

    # Filename for pickle file, this is to save the last GA object
    savefilename = "finalga.pickle"

    # Cores for multiprocessing
    cores = 4

    # Manipulator parameters
    desired_position = [5, 5, 5]
    manipulator_dimensions = [5, 5, 5, 5]
    manipulator_mass = [1, 1, 1]

    # Permanent Configurations
    json_config_folder = "JSON files"
    prev_parameters_dir = json_config_folder + "/runs_parameters.json"
    quick_models_dir = json_config_folder + "/quick_models.json"

    ## Execution

    manipulator = RoboticManipulator(manipulator_dimensions, manipulator_mass)

    save_load_json = JSONSaveLoad(parameters_from_filename=prev_parameters_dir,
                                  quick_save_filename=quick_models_dir,
                                  save_filename=run_name,
                                  parameters_variations=parameters_variations)

    if all:
        if from_file:
            save_load_json.loadParameters()

            runs = save_load_json.getRuns()
            print(f"Executing models on {cores} cores...")
            executer = MultiCoreExecuter(runs, manipulator, save_load_json, cores=cores)
            executer.run()

        else:
            GA.runAlgorithm()
            best_individual = GA.getBestIndividual()

            if input("Save Json? [Y/N]") == "Y":
                save_load_json.saveJson(GA, quick=True)
            elif input("Sure? [Y/N]") == "N":
                save_load_json.saveJson(GA, quick=True)

            with open(savefilename, 'wb') as f:
                pickle.dump(GA, f, pickle.HIGHEST_PROTOCOL)
    else:
        GA.initialization()
        individual = GA.getPopulation()[0]
        for i, ang in enumerate(np.transpose(individual.getGenes())):
            plt.plot(ang)

        plt.legend([r"$\theta_1$", r"$\theta_2$", r"$\theta_3$", r"$\theta_4$"])
        plt.xlabel("Unidad de tiempo")
        plt.ylabel("Radianes")

        plt.show()

        with open("gasafe.pickle", "rb") as f:
            ga = pickle.load(f)
            ga.graph(2)
            ind = ga.getBestIndividual()
            plt.legend([r"$\theta_1$", r"$\theta_2$", r"$\theta_3$", r"$\theta_4$"])
            plt.title("Mejor individuo")
            plt.xlabel("Unidad de Tiempo")
            plt.ylabel("Ángulo [rad]")
            plt.show()