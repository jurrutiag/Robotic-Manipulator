
if __name__ == "__main__":
    import GeneticAlgorithm
    import numpy as np
    import time
    import pickle

    np.random.seed(0)

    ga = GeneticAlgorithm.GeneticAlgorithm(desired_position=[3, 0, 0])

    ga.initialization()

    print("Individual:")
    individual = ga.getPopulation()[0]
    print(individual.getGenes()[:-1])

    final_angle = individual.getFinalAngle()

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
    torques = ff.getTorques(angularAccelerations, inertias, positions)
    print(torques)

    print("Fitness: ")
    t0 = time.time()
    ff.evaluateFitness(individual)
    tf = time.time()
    print(individual.getFitness())
    print(f"Time for evaluation: {tf - t0} seg")

    savefilename = "ga.pickle"

    with open(savefilename, 'wb') as f:
        pickle.dump(ga, f, pickle.HIGHEST_PROTOCOL)
