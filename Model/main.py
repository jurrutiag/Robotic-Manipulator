
def saveJson(jsonfilename, genetic_algorithm):

    new_individual = {}

    with open(jsonfilename) as f:
        individuals_json = json.load(f)
        best_individuals = individuals_json["Best Individuals"]
        new_individual["Genes"] = genetic_algorithm.getBestIndividual().getGenes().tolist()
        new_individual["Info"] = genetic_algorithm.getAlgorithmInfo()
        new_individual["Time of Training"] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
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

    # np.random.seed(0) # for testing

    # True if you want to run all the algorithm, otherwise it will only initialize and show the first individual.
    all = True

    # True if you want to load the parameters from a JSON file
    from_file = True

    # True if you want to fill the JSON file with parameters, this is to easily create the parameters file
    fill_runs_file = True

    # If true, this will empty the runs parameters file
    empty_runs_file = False

    # Filename for pickle file, this is to save the last GA object
    savefilename = "finalga.pickle"

    # Filename for the JSON file where the individuals will go
    json_filename = "JSON files/best_individuals.json"

    # Filename for the JSON file where the parameters are defined
    from_file_filename = "JSON files/runs_parameters.json"

    desired_position = [5, 5, 5]
    manipulator_dimensions = [5, 5, 5, 5]
    manipulator_mass = [1, 1, 1]

    manipulator = RoboticManipulator(manipulator_dimensions, manipulator_mass)
    GA = GeneticAlgorithm(manipulator, desired_position, sampling_points=20)

    if fill_runs_file:
        with open(from_file_filename) as f:
            parameters = json.load(f)
            default_params = parameters["Default"]
            runs = []

            if empty_runs_file:
                parameters["Runs"] = []

            else:
                parameters_variations = {
                    "pop_size": [100, 200],
                    "cross_individual_prob": [0.5, 0.6, 0.7, 0.8],
                    "mut_individual_prob": [0.01, 0.05],
                    "cross_joint_prob": [0.25, 0.5],
                    "mut_joint_prob": [0.25, 0.5],
                    "sampling_points": [20, 30],
                    "torques_error_ponderation": [0, 0.01]
                }

                keys, values = zip(*parameters_variations.items())
                for v in itertools.product(*values):
                    run = default_params.copy()
                    run_change = dict(zip(keys, v))
                    for key, val in run_change.items():
                        run[key] = val
                    runs.append(run)
                parameters["Runs"] = runs

        with open(from_file_filename, 'w') as f:
            json.dump(parameters, f)

    if all:
        if from_file:

            with open(from_file_filename) as f:
                file = json.load(f)
                runs = file["Runs"]
                for i, run in enumerate(runs):
                    print(f"Run number {i}, with parameters: " + json.dumps(run))
                    GA = geneticWrapper(GeneticAlgorithm, manipulator, run)
                    GA.runAlgorithm()
                    saveJson(json_filename, GA)
        else:
            GA.runAlgorithm()
            best_individual = GA.getBestIndividual()

            if input("Save Json? [Y/N]") == "Y":
                saveJson(json_filename, GA)
            elif input("Sure? [Y/N]") == "N":
                saveJson(json_filename, GA)

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