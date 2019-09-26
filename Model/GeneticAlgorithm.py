import math
import numpy as np
import random
import Individual
import RoboticManipulator

class GeneticAlgorithm:

    def __init__(self, pop_size=100, cross_individual_prob=0.5, mut_individual_prob=0.1, cross_joint_prob=0.5, mut_joint_prob=0.5, sampling_points=50, manipulator_dimensions=[1, 1, 1, 1]):

        # Algorithm parameters

        self._pop_size = pop_size
        self._cross_individual_prob = cross_individual_prob
        self._mut_individual_prob = mut_individual_prob
        self._cross_joint_prob = cross_joint_prob
        self._mut_joint_prob = mut_joint_prob
        self._sampling_points = sampling_points # N_k

        # Algorithm variables

        self._population = []

        # Manipulator

        self._manipulator = RoboticManipulator.RoboticManipulator(manipulator_dimensions)

    def initialization(self, populationSize, initialAngles, finalAngles, armPivots, steps):
        P = np.zeros((steps, armPivots))
        results = []
        for ind in range(populationSize):
            for h in range(armPivots):
                #solo el extremo inicial esta fijo
                average = random.randrange(2, steps) 
                std = random.randrange(1,steps/6)
                R= abs(initialAngles[h]-finalAngles[h])
            
                for i in range(steps):
                    #no estoy seguro si habra que poner step distinto
                    noise = random.randrange(-3*R, 3*R)*math.exp(-(i-average)**2/(2*std**2))       
                    P[i,h] = initialAngles[h] + (i-1)*(initialAngles[h]*finalAngles[h])/(steps-1)+noise


            results.append(P)

        #lista de individuos
        return results

    def angleCorrection(self, minAngles, maxAngles):

        for ind in self._population:
            for i in ind.shape(0):
                for h in shape(1):
                    dif = abs(ind[i,h]-maxAngles[h])
                    if ind[i,h]>maxAngles[h]:
                        ind[i,h] =maxAngles[h] - dif
                    elif ind[i,h]<minAngles[h]:
                        ind[i,h] =minAngles[h] + dif


