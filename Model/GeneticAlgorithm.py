import math
import numpy as np
import random
import Individual
import RoboticManipulator
import FitnessFunction

class GeneticAlgorithm:

    def __init__(self, desired_position, pop_size=100, cross_individual_prob=0.5, mut_individual_prob=0.1, cross_joint_prob=0.5, mut_joint_prob=0.5, sampling_points=50, manipulator_dimensions=[1, 1, 1, 1], manipulator_mass=[1, 1, 1, 1], torques_ponderations=[1, 1, 1, 1]):

        # Algorithm parameters

        self._pop_size = pop_size
        self._cross_individual_prob = cross_individual_prob
        self._mut_individual_prob = mut_individual_prob
        self._cross_joint_prob = cross_joint_prob
        self._mut_joint_prob = mut_joint_prob
        self._sampling_points = sampling_points # N_k

        # Algorithm variables

        self._population = []
        self._parents = []
        self._children = []

        # Manipulator

        self._manipulator = RoboticManipulator.RoboticManipulator(manipulator_dimensions, manipulator_mass)

        # Fitness Function

        self._fitness_function = FitnessFunction.FitnessFunction(self._manipulator, torques_ponderations, desired_position)

    def initialization(self, initialAngles=[0, 0, 0, 0]):

        P = np.zeros((self._sampling_points, 4))
        results = []
        for ind in range(self._pop_size):

            finalAngles = np.pi * np.random.random(size=4) - np.pi/2

            for h in range(4):
                #solo el extremo inicial esta fijo
                average = (self._sampling_points - 2) * np.random.random() + self._sampling_points
                # average = random.randrange(2, self._sampling_points)
                std = (self._sampling_points / 6 - 1) * np.random.random() + 1
                # std = random.randrange(1,self._sampling_points/6)
                R = abs(initialAngles[h] - finalAngles[h])
            
                for i in range(self._sampling_points):
                    #no estoy seguro si habra que poner step distinto
                    noise = (6*R) *  np.random.random()*math.exp(-(i-average)**2/(2*std**2)) - 3 * R
                    P[i,h] = initialAngles[h] + (i-1)*(finalAngles[h] - initialAngles[h])/(self._sampling_points-1) + noise


            results.append(Individual.Individual(P))

        #lista de individuos
        self._population = results

    def getPopulation(self):
        return self._population

    def getManipulator(self):
        return self._manipulator

    def angleCorrection(self, minAngles, maxAngles):

        for ind in self._population:
            ind_genes = ind.getGenes()
            for i in range(ind_genes.shape[0]):
                for h in range(ind_genes.shape[1]):
                    dif = abs(ind_genes[i, h] - maxAngles[h])
                    if ind_genes[i, h] > maxAngles[h]:
                        ind_genes[i, h] = maxAngles[h] - dif
                    elif ind_genes[i, h] < minAngles[h]:
                        ind_genes[i, h] = minAngles[h] + dif


    #probabilidades de la selección
    def probabilitiesOfSelection(self, fitnessValues):
        total = 0
        for fitness in fitnessValues:
            total += fitness
        probabilities = []
        for fitness in fitnessValues:
            probabilities.append(fitness/total)

        return probabilities


    def selection(self, rate, probabilities):
        parents=[]
        amountOfParents = int(rate*len(self._population))

        i=0
        while amountOfParents!=0:
            if(probabilities[i]>random.random()):
                amountOfParents-=1
                parents.append(self._population[i])
            i+=1

        self._parents = parents


    def mutation(self, average, std):
        for ind in len(self._children):
            ind_mat = ind.getGenes()
            if self._mut_individual_prob<random.random():
                continue

            for h in range(4):
                if self._mut_joint_prob<random.random():
                    continue

                ## Diferencia entre valores menores y mayores del hijo que se esta mutando.
                # R= abs(initialAngles[h]-finalAngles[h])
            
                for i in range(self._sampling_points):
                    # d = random.randrange(-R, R)
                    # ind_mat[i,h] = ind_mat[i,h] + d**math.exp(-(i-average)**2/(2*std**2))
                    pass

            ind.setGenes(ind_mat)


    def getFitnessFunction(self):
        return self._fitness_function


            


