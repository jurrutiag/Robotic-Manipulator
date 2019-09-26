
if __name__ == "__main__":
    import GeneticAlgorithm
    import numpy as np

    np.random.seed(0)

    ga = GeneticAlgorithm.GeneticAlgorithm(desired_position=[3, 0, 0])

    ga.initialization(np.random.randint(50, size=4), np.random.randint(50, size=4))

    print("Individual:")
    individual = ga.getPopulation()[0]
    print(individual.getGenes()[:-1])

    ff = ga.getFitnessFunction()

    print("Positions:")
    positions = ff.getPositions(individual)
    print(positions)

    print("Accelerations: ")
    angularAccelerations = ff.getAngularAccelerations(individual)
    print(angularAccelerations)

    print("Inertias: ")
    inertias = ff.getInertias(positions)
    print(inertias)

    print("Torques: ")
    torques = ff.getTorques(angularAccelerations, inertias)
    print(torques)

    print("Fitness: ")
    ff.evaluateFitness(individual)
    print(individual.getFitness())