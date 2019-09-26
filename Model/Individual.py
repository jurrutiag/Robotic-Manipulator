

class Individual:

    def __init__(self, genes):
        self._genes = genes
        self._fitness = None

    def getGenes(self):
        return self._genes

    def setGenes(self, genes):
        self._genes = genes

    def getMutationProb(self):
        return self._mutation_prob

    def getFitness(self):
        return self._fitness

    def setFitness(self, fitness):
        self._fitness = fitness

    def getFinalAngle(self):
        return self._genes[-1]