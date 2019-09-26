
if __name__ == "__main__":
    import GeneticAlgorithm
    import FitnessFunction


    ga = GeneticAlgorithm.GeneticAlgorithm()

    ga.initialization([0, 0, 0, 0], [50, 50, 50, 50])
    print(ga.getPopulation()[0].getGenes())