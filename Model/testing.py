
if __name__ == "__main__":
    import GeneticAlgorithm
    import FitnessFunction


    ga = GeneticAlgorithm.GeneticAlgorithm()

    ga.initialization([20, 20, 20, 20], [50, 50, 50, 50])
    print(ga.getPopulation()[0].getGenes())

    print(FitnessFunction.getPositions(ga.getPopulation(), ga.getManipulator())[0])

    FitnessFunction.getAccelerations(FitnessFunction.getPositions(ga.getPopulation(), ga.getManipulator()))