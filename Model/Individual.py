

class Individual:

    def __init__(self, genes, id=0):
        self._genes = genes
        self._fitness = None
        self._torque = None
        self._distance = None
        self._id = id

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

    def setId(self, id):
        self._id = id

    def getId(self):
        return self._id

    def setTorque(self, torque):
        self._torque = torque

    def getTorque(self):
        return self._torque

    def setDistance(self, distance):
        self._distance = distance

    def getDistance(self):
        return self._distance