import math
import numpy as np
import matplotlib.pyplot as plt
import random
import Individual
import FitnessFunction
import time
import pickle
from PrintModule import PrintModule
import multiprocessing


class GeneticAlgorithm:

    def __init__(self, manipulator, desired_position, print_module=None, cores=1, pop_size=100, cross_individual_prob=0.6,
                 mut_individual_prob=0.05, cross_joint_prob=0.5, mut_joint_prob=0.5, pairing_prob=0.5,
                 sampling_points=20, torques_ponderations=(1, 1, 1, 1), generation_threshold=3000,
                 fitness_threshold=0.8, progress_threshold=1, generations_progress_threshold=50,
                 torques_error_ponderation=0.01, distance_error_ponderation=1, rate_of_selection=0.3, elitism_size=20,
                 selection_method="rank", rank_probability=0.5, generation_for_print=10, safe_save=True,
                 plot_fitness=True, plot_best=False, exponential_initialization=False, total_time=5):

        # Algorithm info for save

        self._all_info = locals().copy()
        del self._all_info["print_module"]
        del self._all_info["manipulator"]
        del self._all_info["self"]

        # Algorithm parameters

        self._pop_size = pop_size
        self._cross_individual_prob = cross_individual_prob
        self._mut_individual_prob = mut_individual_prob
        self._cross_joint_prob = cross_joint_prob
        self._mut_joint_prob = mut_joint_prob
        self._pairing_prob = pairing_prob
        self._sampling_points = sampling_points # N_k
        self._initial_angles = [0, 0, 0, 0]

        self._elitism_size = elitism_size
        self._rate_of_selection = rate_of_selection if (rate_of_selection * pop_size >= 2 * elitism_size) else (2 * elitism_size / pop_size)
        self._selection_method = selection_method
        self._rank_probability = rank_probability
        self._safe_save = safe_save
        self._save_filename = "gasafe.pickle"
        self._generation_for_print = generation_for_print
        self._plot_best = plot_best
        self._plot_fitness = plot_fitness
        self._exponential_initialization = exponential_initialization

        # Algorithm variables

        self._population = []
        self._parents = []
        self._children = []

        # Manipulator

        self._manipulator = manipulator

        # Fitness Function

        self._fitness_function = FitnessFunction.FitnessFunction(self._manipulator, torques_ponderations,
                                                                 desired_position, sampling_points,
                                                                 total_time, distance_error_ponderation=distance_error_ponderation,
                                                                 torques_error_ponderation=torques_error_ponderation)

        # Fitness Results

        self._best_case = []
        self._average_case = []
        self._generation = 0

        # Progress Threshold

        self._generation_threshold = generation_threshold
        self._fitness_threshold = fitness_threshold
        self._progress_threshold = progress_threshold
        self._generations_progress_threshold = generations_progress_threshold

        # Algorithm timing, resource measure and prints

        self._print_module = print_module if print_module is not None else PrintModule()

        # Final Results

        self._best_individual = None
        self._graphs = None

        # MultiCore algorithm

        self._cores = cores
        self._in_queue = None
        self._out_queue = None
        if self._cores > 1:

            self._processes = []
            self._in_queue = multiprocessing.Queue()
            self._out_queue = multiprocessing.Queue()

            for i in range(cores - 1):
                p = multiprocessing.Process(target=self.multiCoreFitness, args=(self._in_queue, self._out_queue))
                self._processes.append(p)
                p.start()

    def runAlgorithm(self):

        self._start_time = time.time()

        # First generation gets created
        self.initialization()

        # First generation fitness
        self._population = self.evaluateFitness(self._population)
        self.getBestAndAverage()

        while True:

            # Save for every 100 generations
            if self._generation % 100 == 0 and self._safe_save:
                with open(self._save_filename, 'wb') as f:
                    pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
                    print("| SAVED |")

            # Selection of parents
            self.selection()

            # Children are generated using crossover
            self.generateChildren()

            assert(len(self._children) == self._pop_size)

            # Children are mutated
            self.mutation()

            # Children are evaluated
            self._children = self.evaluateFitness(self._children)

            # Parents are replaced by children
            self.replacement()
            self.getBestAndAverage()

            # Correct angles out of the range
            self.angleCorrection()

            # Check for termination condition
            if self.terminationCondition():
                self._total_training_time = time.time() - self._start_time
                self.findBestIndividual()
                if self._plot_best:
                    self.plotBest()
                self.printGenerationData()
                if self._plot_fitness:
                    self.graph(2)
                if self._cores > 1:
                    self.buryProcesses()
                return

            # Information is printed
            if self._generation_for_print and self._generation % self._generation_for_print == 0:
                if self._plot_best:
                    self.plotBest()
                self.printGenerationData()

    def initialization(self):

        results = []

        finalAngles = np.pi * np.random.random(size=(self._pop_size, 4)) - np.pi/2

        for ind in range(self._pop_size):
            P = np.zeros((self._sampling_points, 4))

            for h in range(4):
                # Solo el extremo inicial esta fijo
                average = (self._sampling_points - 2) * np.random.random() + 2
                std = (self._sampling_points / 6 - 1) * np.random.random() + 1

                P[0, h] = self._initial_angles[h]

                if self._exponential_initialization:
                    R = abs(self._initial_angles[h] - finalAngles[ind][h])
                    for i in range(2, self._sampling_points + 1):
                        A = (6 * R) * np.random.random() - 3 * R
                        noise = A * math.exp(-(i - average) ** 2 / (2 * std ** 2))
                        P[i - 1, h] = self._initial_angles[h] + (i - 1) * (finalAngles[ind][h] - self._initial_angles[h]) / (self._sampling_points - 1) + noise

                else:
                    for i in range(2, self._sampling_points + 1):
                        P[i - 1, h] = self._initial_angles[h] + (finalAngles[ind][h] - self._initial_angles[h]) * 0.5 * (1 + np.tanh((i - average) / std))

            results.append(Individual.Individual(P))

        self._generation = 1
        self._population = results

    def evaluateFitness(self, population):
        if self._cores > 1:
            split_population = np.array_split(population, self._cores)
            for mini_pop in split_population[:-1]:
                self._in_queue.put(mini_pop)

            out_list = self.singleCoreFitness(split_population[-1])

            for i in range(self._cores - 1):
                # print(self._out_queue.get())
                out_list = np.concatenate((out_list, self._out_queue.get()))

            assert(len(out_list) == self._pop_size)
            return out_list

        else:
            return self.singleCoreFitness(population)

    def singleCoreFitness(self, population):
        for individual in population:
            self._fitness_function.evaluateFitness(individual)

        return population

    def multiCoreFitness(self, in_queue, out_queue):
        while True:
            pop = in_queue.get()
            if list(pop) == list("finish"):
                break
            out_queue.put(self.singleCoreFitness(pop))

    def angleCorrection(self):
        angle_limits = self._manipulator.getLimits()

        for ind in self._population:
            ind_genes = ind.getGenes()

            for i in range(ind_genes.shape[0]):
                for h in range(ind_genes.shape[1]):
                    maxAngle = angle_limits[h][1]
                    minAngle = angle_limits[h][0]

                    if ind_genes[i, h] > maxAngle:
                        ind_genes[i, h] = maxAngle  # - (ind_genes[i, h] - maxAngle)

                    elif ind_genes[i, h] < minAngle:
                        ind_genes[i, h] = minAngle  # + (minAngle - ind_genes[i,h])

    def selection(self):
        amount_of_parents = int(self._rate_of_selection * len(self._population))

        self._population = sorted(self._population, key=lambda x: x.getFitness(), reverse=True)
        self._children = self._population[:self._elitism_size]

        population_left = self._population[self._elitism_size:]

        if self._selection_method == "fitness_proportional":

            # Probabilities of selection for each individual is calculated
            fitness_values = []
            for individual in population_left:
                fitness_values.append(individual.getFitness())

            total = sum(fitness_values)
            probabilities = []

            for fitness in fitness_values:
                probabilities.append(fitness / total)

            self._parents = np.random.choice(population_left, size=(amount_of_parents - self._elitism_size), p=probabilities)

        # Rank is the default in case of misspelling
        else:
            probabilities = []

            for i, ind in enumerate(population_left):
                i += 1 # Starts from one

                # P_i = P_c (1 - P_c) ^ (i - 1), P_n = (1 - P_c) ^ (n - 1)
                probabilities.append((self._rank_probability if i != self._pop_size else 1) * (1 - self._rank_probability) ** (i - 1))

            self._parents = np.random.choice(population_left, size=(amount_of_parents - self._elitism_size), p=probabilities)

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
        i = 0
        j = 0
        while len(self._children) != self._pop_size:
            if i == amount  and j == amount -1:
                i = 0
                j = 0
                coinToss = np.random.rand(amount, amount)

            if coinToss[i,j] < self._pairing_prob and i != j:
                child1, child2 = self.crossover(self._parents[i], self._parents[j])
                self._children.append(child1)
                self._children.append(child2)
            j += 1
            if j == amount:
                j = 0
                i +=1
            if i == amount:
                j = 0
                i = 0
                coinToss = np.random.rand(amount, amount)

    def mutation(self):

        # Se lanzan todas las monedas antes de iterar
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
        fig_fitness, ax_fitness = plt.subplots(ncols=1, nrows=1)
        cases = ['mejor caso', 'promedio']
        if choice == 0 or choice >= len(cases):
            ax_fitness.plot(self._best_case, label=cases[0])
        if choice == 1 or choice >= len(cases):
            ax_fitness.plot(self._average_case, label=cases[1])

        ax_fitness.legend(["Mejor Caso", "Promedio"])
        ax_fitness.set_xlabel('Generación', fontsize=10)
        ax_fitness.set_ylabel('Función de Fitness', fontsize=10)
        ax_fitness.set_title('Evolución del algoritmo genético')

        fig_best_individual, ax_best_individual = plt.subplots(ncols=1, nrows=1)

        for ang in np.transpose(self._best_individual.getGenes()):
            ax_best_individual.plot(ang)

        ax_best_individual.legend([r"$\theta_1$", r"$\theta_2$", r"$\theta_3$", r"$\theta_4$"])
        ax_best_individual.set_title("Mejor individuo")
        ax_best_individual.set_xlabel("Unidad de Tiempo")
        ax_best_individual.set_ylabel("Ángulo [rad]")

        self._graphs = fig_fitness, fig_best_individual

    def printGenerationData(self):
        t = time.time() - self._start_time

        self._print_module.print("| Generation:                    %4.4d |\n" % (self._generation) +
              "| Best Generation Fitness: %10.8f |\n" % (self._best_case[self._generation - 1]) +
              "| Mean Generation Fitness: %10.8f |\n" % (self._average_case[self._generation - 1]) +
              "| Best Overall Fitness:    %10.8f |\n" % (max(self._best_case)) +
              "| Total time:                  %6.2f |\n" % (t) +
              "- - - - - - - - - - - - - - - - - - - -", position="Current Training")

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

    def buryProcesses(self):

        for i in range(self._cores - 1):
            self._in_queue.put("finish")

        for p in self._processes:
            p.join()

    def getAlgorithmInfo(self):
        return self._all_info

    def getPopulation(self):
        return self._population

    def getManipulator(self):
        return self._manipulator

    def getBestIndividual(self):
        return self._best_individual

    def getbestCase(self):
        return self._best_case

    def getAverageCase(self):
        return self._average_case

    def getGraphs(self):
        return self._graphs

    def getTrainingTime(self):
        return self._total_training_time

    def getGeneration(self):
        return self._generation