
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



if __name__ == "__main__":
    from GeneticAlgorithm import GeneticAlgorithm
    from RoboticManipulator import RoboticManipulator
    import numpy as np
    import pickle
    import matplotlib.pyplot as plt
    import json
    import datetime

    # np.random.seed(0) # for testing

    all = True

    savefilename = "finalga.pickle"
    json_filename = "best_individuals.json"


    desired_position = [5, 5, 5]
    manipulator_dimensions = [5, 5, 5, 5]
    manipulator_mass = [1, 1, 1]

    manipulator = RoboticManipulator(manipulator_dimensions, manipulator_mass)
    GA = GeneticAlgorithm(desired_position, manipulator, sampling_points=20)

    if all:
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