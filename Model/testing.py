
if __name__ == "__main__":
    import GeneticAlgorithm
    import FitnessFunction
    import numpy as np

    np.random.seed(0)

    ga = GeneticAlgorithm.GeneticAlgorithm()

    ga.initialization(np.random.randint(50, size=4), np.random.randint(50, size=4))

    print("Individual:")
    print(ga.getPopulation()[0].getGenes())

    ff = FitnessFunction.FitnessFunction(ga.getManipulator())

    print("Positions:")
    positions = ff.getPositions(ga.getPopulation()[0])
    print(positions)

    print("Accelerations: ")
    angularAccelerations = ff.getAngularAccelerations(ga.getPopulation()[0])
    print(angularAccelerations)

    print("Inertias: ")
    inertias = ff.getInertias(positions)
    print(inertias)

    print("Torques: ")
    torques = ff.getTorques(angularAccelerations, inertias)
    print(torques)