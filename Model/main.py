
if __name__ == "__main__":
    from GeneticAlgorithm import GeneticAlgorithm
    import numpy as np
    import pickle

    np.random.seed(0) # for testing

    desired_position = [3, 3, 3]
    manipulator_dimensions = [5, 5, 5, 5]
    manipulator_mass = [1, 1, 1]

    RoboticManipulator.RoboticManipulator(manipulator_dimensions, manipulator_mass)
    GA = GeneticAlgorithm(desired_position)

    GA.runAlgorithm()

    best_individual = GA.getBestIndividual()

    savefilename = "ga.pickle"

    with open(savefilename, 'wb') as f:
        pickle.dump(GA, f, pickle.HIGHEST_PROTOCOL)
