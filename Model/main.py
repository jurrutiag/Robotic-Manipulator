
if __name__ == "__main__":
    from GeneticAlgorithm import GeneticAlgorithm
    from RoboticManipulator import RoboticManipulator
    import numpy as np
    import pickle
    import matplotlib.pyplot as plt

    # np.random.seed(0) # for testing

    desired_position = [0, 0, 20]
    manipulator_dimensions = [5, 5, 5, 5]
    manipulator_mass = [1, 1, 1]

    manipulator = RoboticManipulator(manipulator_dimensions, manipulator_mass)
    GA = GeneticAlgorithm(desired_position, manipulator, sampling_points=50)

    GA.runAlgorithm()
    best_individual = GA.getBestIndividual()

    # GA.initialization()
    # individual = GA.getPopulation()[0]
    # for i, ang in enumerate(np.transpose(individual.getGenes())):
    #     plt.plot(ang)
    #
    # plt.legend([r"$\theta_1$", r"$\theta_2$", r"$\theta_3$", r"$\theta_4$"])
    # plt.xlabel("Unidad de tiempo")
    # plt.ylabel("Radianes")
    #
    # plt.show()

    savefilename = "ga.pickle"

    with open(savefilename, 'wb') as f:
        pickle.dump(GA, f, pickle.HIGHEST_PROTOCOL)
