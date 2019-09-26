

class Individual:

    def __init__(self, genes):
        self._genes = genes

    def getGenes(self):
        return self._genes

    def setGenes(self, genes):
        self._genes = genes

    def getMutationProb(self):
        return self._mutation_prob