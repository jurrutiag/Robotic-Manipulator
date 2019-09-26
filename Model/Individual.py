

class Individual:

    def __init__(self, genes, mutation_prob):
        self._genes = genes
        self._mutation_prob = mutation_prob

    def getGenes(self):
        return self._genes

    def setGenes(self, genes):
        self._genes = genes

    def getMutationProb(self):
        return self._mutation_prob