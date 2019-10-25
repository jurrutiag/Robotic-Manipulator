

class Individual:

    def __init__(self, genes, id=0):
        self._genes = genes
        self._fitness = None
        self._multi_fitness = None
        self._id = id

    def getGenes(self):
        return self._genes

    def setGenes(self, genes):
        self._genes = genes

    def getMutationProb(self):
        return self._mutation_prob

    def getMultiFitness(self):
        return self._multi_fitness

    def setMultiFitness(self, multi_fitness):
        self._multi_fitness = multi_fitness

    def getFitness(self):
        return self._fitness

    def setFitness(self, fitness):
        self._fitness = fitness

    def getFinalAngle(self):
        return self._genes[-1]

    def setId(self, id):
        self._id = id

    def getId(self):
        return self._id

    def __str__(self):
        return f"Individual {id(self)}, Fitness: {self._fitness}"