import math
import numpy as np
import matplotlib.pyplot as plt
import random
import Individual
import RoboticManipulator
import FitnessFunction

class GeneticAlgorithm:

    def __init__(self, desired_position, pop_size=100, cross_individual_prob=0.5, mut_individual_prob=0.1, cross_joint_prob=0.5, mut_joint_prob=0.5, pairing_prob=0.5, sampling_points=50, manipulator_dimensions=[1, 1, 1, 1], manipulator_mass=[1, 1, 1, 1], torques_ponderations=[1, 1, 1, 1]):

        # Algorithm parameters

        self._pop_size = pop_size
        self._cross_individual_prob = cross_individual_prob
        self._mut_individual_prob = mut_individual_prob
        self._cross_joint_prob = cross_joint_prob
        self._mut_joint_prob = mut_joint_prob
        self._pairing_prob = pairing_prob
        self._sampling_points = sampling_points # N_k
        self._initial_angles = [0, 0, 0, 0]
        self._rate_of_selection = 0.7

        # Algorithm variables

        self._population = []
        self._parents = []
        self._children = []

        # Manipulator

        self._manipulator = RoboticManipulator.RoboticManipulator(manipulator_dimensions, manipulator_mass)

        # Fitness Function

        self._fitness_function = FitnessFunction.FitnessFunction(self._manipulator, torques_ponderations, desired_position)

        # Fitness Results

        self._best_case = []
        self._average_case = []
        self._generation = 0

        # Final Results

        self._best_individual = None

    def runAlgorithm(self):

        # First generation gets created
        self.initialization()

        # First generation fitness
        self.evaluateFitness(self._population)

        while True:
            # Probabilities of selection for each individual is calculated
            fitness_values = []
            for individual in self._population:
                fitness_values.append(individual.getFitness())
            probabilities = self.probabilitiesOfSelection(fitness_values)

            # Selection of parents
            self.selection(self._rate_of_selection, probabilities)

            # Children are generated using crossover
            self.generateChildren()

            # Children are mutated
            self.mutation()

            # Children are evaluated
            self._fitness_function.evaluateFitness(self._children)

            # Parents are replaced by children
            self.replacement()

            # Check for termination condition
            if self.terminationCondition():
                self.getBestIndividual()
                return

    def initialization(self):

        P = np.zeros((self._sampling_points, 4))
        results = []

        finalAngles = np.pi * np.random.random(size=(self._pop_size, 4)) - np.pi/2

        for ind in range(self._pop_size):

            # finalAngles = np.pi * np.random.random(size=4) - np.pi/2

            for h in range(4):
                #solo el extremo inicial esta fijo
                average = (self._sampling_points - 2) * np.random.random() + self._sampling_points
                # average = random.randrange(2, self._sampling_points)
                std = (self._sampling_points / 6 - 1) * np.random.random() + 1
                # std = random.randrange(1,self._sampling_points/6)
                R = abs(self._initial_angles[h] - finalAngles[ind][h])

                for i in range(self._sampling_points):
                    #no estoy seguro si habra que poner step distinto
                    noise = (6*R) *  np.random.random()*math.exp(-(i-average)**2/(2*std**2)) - 3 * R
                    P[i,h] = self._initial_angles[h] + (i-1)*(finalAngles[ind][h] - self._initial_angles[h])/(self._sampling_points-1) + noise


            results.append(Individual.Individual(P))

        #lista de individuos
        self._generation = 1
        self._population = results

    def evaluateFitness(self, population):

        for individual in population:
            self._fitness_function.evaluateFitness(individual)

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

    def crossover(self, ind1, ind2):

        mu = (self._sampling_points - 1) * np.random.random() + 1
        std = (self._sampling_points / 6 - 1) * np.random.random() + 1

        gene_1 = ind1.getGenes()
        gene_2 = ind2.getGenes()

        child_1_genes = np.zeros((self._sampling_points, 4))
        child_2_genes = np.zeros((self._sampling_points, 4))

        for i in range(self._sampling_points):
            w = 0.5 * (1 + np.tanh((i - mu) / std))
            for h in range(4):
                child_1_genes[i, h] = w * gene_1[i, h] + (1 - w) * gene_2[i, h]
                child_2_genes[i, h] = (1 - w) * gene_1[i, h] + w * gene_2[i, h]

        return Individual.Individual(child_1_genes), Individual.Individual(child_2_genes)

    def generateChildren(self):
        amount=len(self._parents)
        coinToss = np.random.rand(amount, amount)
        for i in range(amount):
            for j in range(amount):

                if coinToss[i,j] < self._pairing_prob:
                    child1, child2 = crossover(self._parents[i],self._parents[j])
                    self._children.append(child1)
                    self._children.append(child2)
                if len(self._children) == self._pop_size:
                    return






    def mutation(self):

        mu = (self._sampling_points - 1) * np.random.random() + 1
        std = (self._sampling_points / 6 - 1) * np.random.random() + 1

        #se lanzan todas las monedas antes de iterar
        coin_toss_ind = np.random.rand(1, len(self._children))
        coin_toss_joint = np.random.rand(4, len(self._children))

        for ind in range(len(self._children)):
            ind_mat = self._children[ind].getGenes()
            if self._mut_individual_prob < coin_toss_ind[ind]:
                continue

            for h in range(4):
                if self._mut_joint_prob < coin_toss_joint[h, ind]:
                    continue

                ## Diferencia entre valores menores y mayores del hijo que se esta mutando.
                R = np.max(ind_mat[:, h]) - np.min(ind_mat[:, h])

                d = np.random.rand(1, self._sampling_points) * 2 * R - R
                for i in range(self._sampling_points):
                    ind_mat[i, h] = ind_mat[i, h] + d[i] * math.exp((- (i - mu) ** 2) / (2 * std ** 2))

            self._children[ind].setGenes(ind_mat)

    def replacement(self):
        self._population = self._children
        self._parents = []
        self._children = []

    def terminationCondition(self):
        pass

    def getBestIndividual(self):
        fit = 0
        for individual in self._population:
            if individual.getFitness() >= fit:
                self._best_individual = individual
                fit = individual.getFitness()

    def getFitnessFunction(self):
        return self._fitness_function


    def getBestAndAverage(self):
        minFitness = 0
        meanFitness = 0
        for ind in self._population:
            fitness = ind.getFitnessFunction
            minFitness = fitness if fitness < minFitness else minFitness
            meanFitness += fitness
        meanFitness /= len(self._population)

        self._best_case.append(minFitness)
        self._average_case.append(meanFitness)



    def graph(self, choice):
        axes=fig.add_subplot(111)
        cases = ['mejor caso', 'promedio']
        if choice == 0 or choice > len(cases):
            plt.plot(range(self._generation)+1, self._best_case, label = cases[i])
        if choice == 1 or choice > len(cases):
            plt.plot(range(self._generation)+1, self._average_case, label = cases[i])

        plt.xlabel('Generación', fontsize=10)
        plt.ylabel('Función de Fitness', fontsize=10)
        plt.suptitle('Evolución del algoritmo genético')
        plt.show()

    def getPopulation(self):
        return self._population

    def getManipulator(self):
        return self._manipulator



