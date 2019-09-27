import math
import numpy as np
import matplotlib.pyplot as plt
import random
import Individual
import RoboticManipulator
import FitnessFunction
import time
import pickle

class GeneticAlgorithm:

    def __init__(self, desired_position, manipulator, pop_size=100, cross_individual_prob=0.6, mut_individual_prob=0.01, cross_joint_prob=0.5, mut_joint_prob=0.5, pairing_prob=0.5, sampling_points=50, torques_ponderations=(1, 1, 1, 1), generation_threshold = 200, fitness_threshold = 1, progress_threshold = 1, generations_progress_threshold = 50):

        # Algorithm parameters

        self._pop_size = pop_size
        self._cross_individual_prob = cross_individual_prob
        self._mut_individual_prob = mut_individual_prob
        self._cross_joint_prob = cross_joint_prob
        self._mut_joint_prob = mut_joint_prob
        self._pairing_prob = pairing_prob
        self._sampling_points = sampling_points # N_k
        self._initial_angles = [0, 0, 0, 0]
        self._rate_of_selection = 0.3
        self._safe_save = True
        self._save_filename = "gasafe.pickle"

        # Algorithm variables

        self._population = []
        self._parents = []
        self._children = []

        # Manipulator

        self._manipulator = manipulator

        # Fitness Function

        self._fitness_function = FitnessFunction.FitnessFunction(self._manipulator, torques_ponderations, desired_position, torques_error_ponderation=1)

        # Fitness Results

        self._best_case = []
        self._average_case = []
        self._generation = 0

        # Progress Threshold

        self._generation_threshold = generation_threshold
        self._fitness_threshold = fitness_threshold
        self._progress_threshold = progress_threshold
        self._generations_progress_threshold = generations_progress_threshold

        # Algorithm timing

        self._start_time = 0

        # Final Results

        self._best_individual = None

    def runAlgorithm(self):

        self._start_time = time.time()

        # First generation gets created
        self.initialization()

        # First generation fitness
        self.evaluateFitness(self._population)
        self.getBestAndAverage()

        while True:

            if self._generation % 20 == 0 and self._safe_save:
                with open(self._save_filename, 'wb') as f:
                    pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
                    print("| SAVED |")

            self.plotBest()
            self.printGenerationData()

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
            self.evaluateFitness(self._children)

            # Parents are replaced by children
            self.replacement()
            self.getBestAndAverage()

            # Correct angles out of the range
            self.angleCorrection()

            # Check for termination condition
            if self.terminationCondition():
                self.findBestIndividual()
                self.graph(2)
                return

    def initialization(self):

        results = []

        finalAngles = np.pi * np.random.random(size=(self._pop_size, 4)) - np.pi/2

        for ind in range(self._pop_size):
            P = np.zeros((self._sampling_points, 4))
            # finalAngles = np.pi * np.random.random(size=4) - np.pi/2

            for h in range(4):
                #solo el extremo inicial esta fijo
                average = (self._sampling_points - 2) * np.random.random() + 2
                # average = random.randrange(2, self._sampling_points)
                std = (self._sampling_points / 6 - 1) * np.random.random() + 1
                # std = random.randrange(1,self._sampling_points/6)
                R = abs(self._initial_angles[h] - finalAngles[ind][h])

                P[0, h] = self._initial_angles[h]
                for i in range(2, self._sampling_points + 1):
                    #no estoy seguro si habra que poner step distinto
                    # A = (6 * R) * np.random.random() - 3 * R
                    # noise = A * math.exp(-(i - average) ** 2 / (2 * std ** 2))
                    # P[i - 1,h] = self._initial_angles[h] + (i - 1) * (finalAngles[ind][h] - self._initial_angles[h])/(self._sampling_points-1) + noise
                    P[i - 1, h] = self._initial_angles[h] + (finalAngles[ind][h] - self._initial_angles[h]) * 0.5 * (1 + np.tanh((i - average) / std))


            results.append(Individual.Individual(P))


        #lista de individuos
        self._generation = 1
        self._population = results

    def evaluateFitness(self, population):

        for individual in population:
            self._fitness_function.evaluateFitness(individual)

    def angleCorrection(self):
        angleLimits = self._manipulator.getLimits()
        for ind in self._population:
            ind_genes = ind.getGenes()
            for i in range(ind_genes.shape[0]):
                for h in range(ind_genes.shape[1]):
                    maxAngle = angleLimits[h][1]
                    minAngle = angleLimits[h][0]
                    if ind_genes[i, h] > maxAngle:
                        ind_genes[i, h] = maxAngle #- (ind_genes[i,h]-maxAngle)
                    elif ind_genes[i, h] < minAngle:
                        ind_genes[i, h] = minAngle #+ (minAngle - ind_genes[i,h])

    #probabilidades de la selección
    def probabilitiesOfSelection(self, fitness_values):
        total = sum(fitness_values)
        probabilities = []

        for fitness in fitness_values:
            probabilities.append(fitness/total)

        return probabilities

    def selection(self, rate, probabilities):
        amount_of_parents = int(rate * len(self._population))

        self._parents = np.random.choice(self._population, size=amount_of_parents, p=probabilities)

    def crossover(self, ind1, ind2):

        mu = (self._sampling_points - 1) * np.random.random() + 1
        std = (self._sampling_points / 6 - 1) * np.random.random() + 1

        gene_1 = ind1.getGenes()
        gene_2 = ind2.getGenes()

        child_1_genes = np.zeros((self._sampling_points, 4))
        child_2_genes = np.zeros((self._sampling_points, 4))

        child_1_genes[0] = self._initial_angles
        child_2_genes[0] = self._initial_angles

        for i in range(2, self._sampling_points + 1):
            w = 0.5 * (1 + np.tanh((i - mu) / std))
            for h in range(4):
                child_1_genes[i - 1, h] = w * gene_1[i - 1, h] + (1 - w) * gene_2[i - 1, h]
                child_2_genes[i - 1, h] = (1 - w) * gene_1[i - 1, h] + w * gene_2[i - 1, h]

        return Individual.Individual(child_1_genes), Individual.Individual(child_2_genes)

    def generateChildren(self):
        amount = len(self._parents)
        coinToss = np.random.rand(amount, amount)

        for i in range(amount):
            for j in range(amount):

                if coinToss[i,j] < self._pairing_prob and i != j:
                    child1, child2 = self.crossover(self._parents[i], self._parents[j])
                    self._children.append(child1)
                    self._children.append(child2)
                if len(self._children) == self._pop_size:
                    return

                if i == amount - 1 and j == amount -1:
                    i=0
                    j=0

    def mutation(self):


        #se lanzan todas las monedas antes de iterar
        coin_toss_ind = np.random.rand(len(self._children))
        coin_toss_joint = np.random.rand(4, len(self._children))

        for ind in range(len(self._children)):

            mu = (self._sampling_points - 1) * np.random.random() + 1
            std = (self._sampling_points / 6 - 1) * np.random.random() + 1

            ind_mat = self._children[ind].getGenes()
            if self._mut_individual_prob < coin_toss_ind[ind]:
                continue


            for h in range(4):
                if self._mut_joint_prob < coin_toss_joint[h, ind]:
                    continue
                low_limit = self._manipulator.getLimits()[h][0]
                high_limit = self._manipulator.getLimits()[h][1]
                ## Diferencia entre valores menores y mayores del hijo que se esta mutando.
                R = high_limit - low_limit

                d = np.random.random() * 2 * R - R

                for i in range(2, self._sampling_points + 1):
                    ind_mat[i - 1, h] = ind_mat[i - 1, h] + d * math.exp((- (i - mu) ** 2) / (2 * std ** 2))

            self._children[ind].setGenes(ind_mat)

    def replacement(self):
        self._population = self._children[:]
        random.shuffle(self._population)
        self._parents = []
        self._children = []
        self._generation += 1

    def terminationCondition(self):
        generationLimitCondition = self._generation > self._generation_threshold
        bestIndividualCondition = self._best_case[len(self._best_case) - 1] > self._fitness_threshold
        # progressCondition = self._best_case[len(self._best_case) - 1 - self._generations_progress_threshold] - self._best_case[len(self._best_case) - 1] < self._progress_threshold
        progressCondition = False

        return generationLimitCondition or bestIndividualCondition or progressCondition



    def findBestIndividual(self):
        fit = 0
        for individual in self._population:
            if individual.getFitness() >= fit:
                self._best_individual = individual
                fit = individual.getFitness()

    def getFitnessFunction(self):
        return self._fitness_function


    def getBestAndAverage(self):
        max_fitness = 0
        mean_fitness = 0
        for ind in self._population:
            fitness = ind.getFitness()
            max_fitness = fitness if fitness > max_fitness else max_fitness
            mean_fitness += fitness
        mean_fitness /= len(self._population)

        self._best_case.append(max_fitness)
        self._average_case.append(mean_fitness)



    def graph(self, choice):
        fig = plt.figure()
        axes = fig.add_subplot(111)
        cases = ['mejor caso', 'promedio']
        if choice == 0 or choice > len(cases):
            plt.plot(range(self._generation + 1), self._best_case, label = cases[choice])
        if choice == 1 or choice > len(cases):
            plt.plot(range(self._generation + 1), self._average_case, label = cases[choice])

        plt.xlabel('Generación', fontsize=10)
        plt.ylabel('Función de Fitness', fontsize=10)
        plt.suptitle('Evolución del algoritmo genético')
        plt.show()

    def printGenerationData(self):
        t = time.time() - self._start_time
        print(f"| Generation: {self._generation}| Best Generation Fitness: {self._best_case[self._generation - 1]} | Mean Generation Fitness: {self._average_case[self._generation - 1]} | Best Overall Fitness: {max(self._best_case)} | Total time: {t} |")

    def plotBest(self):
        fit = 0
        best = None
        for ind in self._population:
            if ind.getFitness() > fit:
                fit = ind.getFitness()
                best = ind
        for ang in np.transpose(best.getGenes()):
            plt.plot(ang)
        plt.show()

    def getPopulation(self):
        return self._population

    def getManipulator(self):
        return self._manipulator

    def getBestIndividual(self):
        return self._best_individual


