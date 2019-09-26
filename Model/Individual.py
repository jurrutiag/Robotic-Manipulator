

class Individual:

    def __init__(self, genes):
        self._genes = genes
        self._mutation_prob = 0

    def getGenes(self):
        return self._genes

    def setGenes(self, genes):
        self._genes = genes
