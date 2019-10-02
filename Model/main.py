
def saveJson(jsonfilename, genetic_algorithm):

    new_individual = {}

    with open(jsonfilename) as f:
        individuals_json = json.load(f)
        best_individuals = individuals_json["Best Individuals"]
        new_individual["Genes"] = genetic_algorithm.getBestIndividual().getGenes().tolist()
        new_individual["Info"] = genetic_algorithm.getAlgorithmInfo()
        new_individual["Time of Training"] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        new_individual["Fitness"] = genetic_algorithm.getBestIndividual().getFitness()
        new_individual["Animate"] = False
        best_individuals.append(new_individual)

    with open(jsonfilename, 'w') as f:
        json.dump(individuals_json, f)


def geneticWrapper(func, manipulator, kwargs):
    return func(manipulator, **kwargs)



if __name__ == "__main__":
    from GeneticAlgorithm import GeneticAlgorithm
    from RoboticManipulator import RoboticManipulator
    import numpy as np
    import pickle
    import matplotlib.pyplot as plt
    import json
    import datetime
    import itertools
    from JSONSaveLoad import JSONSaveLoad
    from MultiCoreExecuter import MultiCoreExecuter

    # np.random.seed(0) # for testing

    # True if you want to run all the algorithm, otherwise it will only initialize and show the first individual.
    all = True

    # True if you want to load the parameters from a JSON file
    from_file = True

    # If true, this will empty the runs parameters file
    empty_runs_file = False

    # Filename for pickle file, this is to save the last GA object
    savefilename = "finalga.pickle"

    # Cores for multiprocessing
    cores = 1

    desired_position = [5, 5, 5]
    manipulator_dimensions = [5, 5, 5, 5]
    manipulator_mass = [1, 1, 1]

    manipulator = RoboticManipulator(manipulator_dimensions, manipulator_mass)
    GA = GeneticAlgorithm(manipulator, desired_position, sampling_points=20)

    parameters_variations = {
        "torques_error_ponderation": [0],
        "pop_size": [100, 200],
        "cross_individual_prob": [0.5, 0.6, 0.7, 0.8],
        "mut_individual_prob": [0.01, 0.05],
        "cross_joint_prob": [0.25, 0.5],
        "mut_joint_prob": [0.25, 0.5],
        "sampling_points": [20, 30]
    }


    save_load_json = JSONSaveLoad(parameters_from_filename="JSON files/runs_parameters.json",
                                  parameters_visualization_filename="JSON files/parameters_visualization.json",
                                  quick_save_filename="JSON files/quick_models.json",
                                  save_filename="json_test", parameters_variations=parameters_variations)

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