import math
import numpy as np
import matplotlib.pyplot as plt
import Individual
import FitnessFunction
import time
import pickle
from PrintModule import PrintModule
import multiprocessing
import pygmo as pg
from mpl_toolkits.mplot3d import Axes3D


class GeneticAlgorithm:

    def __init__(self, manipulator,
                 desired_position,
                 print_module=None,
                 cores=1,
                 pop_size=100,
                 cross_individual_prob=0.6,
                 mut_individual_prob=0.5,
                 cross_joint_prob=0.5,
                 mut_joint_prob=0.5,
                 pairing_prob=0.5,
                 sampling_points=20,
                 torques_ponderations=(1, 1, 1, 1),
                 generation_threshold=1000,
                 fitness_threshold=0.8,
                 progress_threshold=1,
                 generations_progress_threshold=50,
                 torques_error_ponderation=0.0003,
                 distance_error_ponderation=1,
                 velocity_error_ponderation=0.1,
                 rate_of_selection=0.3,
                 elitism_size=10,
                 selection_method=[0, 1, 0, 0],
                 rank_probability=0.5,
                 pareto_tournament_size=5,
                 niche_sigma=100,
                 generation_for_print=10,
                 exponential_initialization=False,
                 total_time=5,
                 individuals_to_display=5):

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
        self._rate_of_selection = rate_of_selection
        self._pareto_tournament_size = pareto_tournament_size
        self._niche_sigma = niche_sigma
        assert(sum(selection_method) == 1)
        self._selection_method = selection_method
        self._rank_probability = rank_probability
        self._generation_for_print = generation_for_print
        self._exponential_initialization = exponential_initialization

        self._individuals_to_display = (np.linspace(1, generation_threshold, individuals_to_display, dtype=int) if individuals_to_display >= 2 else [pop_size])

        # Algorithm variables

        self._population = []
        self._parents = []
        self._children = []
        self._elite = []

        # Manipulator

        self._manipulator = manipulator

        # Fitness Function

        self._fitness_function = FitnessFunction.FitnessFunction(manipulator=self._manipulator,
                                                                 torques_ponderations=torques_ponderations,
                                                                 desired_position=desired_position,
                                                                 sampling_points=sampling_points,
                                                                 total_time=total_time,
                                                                 distance_error_ponderation=distance_error_ponderation,
                                                                 torques_error_ponderation=torques_error_ponderation,
                                                                 velocity_error_ponderation=velocity_error_ponderation)

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
        self._total_training_time = None

        # Final Results

        self._best_individual = None
        self._graphs_fitness = []
        self._graphs_individuals = []
        self._best_individuals_list = []

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

    def runAlgorithm(self, print_data=True):

        self._start_time = time.time()

        # First generation gets created
        self.initialization()

        # First generation fitness
        self._population = self.evaluateFitness(self._population)
        self.getBestAndAverage()

        self.angleCorrection()
        self.findBestIndividual()

        while True:

            if self._generation in self._individuals_to_display:
                self.graphIndividual()

            # Check for termination condition
            if self.terminationCondition():
                self._total_training_time = time.time() - self._start_time
                self.findBestIndividual()

                # Print last information on the terminal
                if print_data:
                    self.printGenerationData()

                self.graph(2)

                # Bury the processes
                if self._cores > 1:
                    self.buryProcesses()

                return

            # Information is printed on the terminal
            if self._generation_for_print and self._generation % self._generation_for_print == 0 and print_data:
                self.printGenerationData()

            if self._generation % 20 == 0:
                #self.plotParetoFrontier()
                pass


            # Selection of parents
            self.selection()

            # Children are generated using crossover
            self.generateChildren()

            # Children are mutated
            self.mutation()

            # Children are evaluated
            self._children = self.evaluateFitness(self._children)

            assert(len(self._children) == self._pop_size - self._elitism_size)

            # Parents are replaced by children
            self.replacement()
            self.getBestAndAverage()

            # Correct angles out of the range
            self.angleCorrection()

            self.findBestIndividual()

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
        split_population = np.array_split(population, self._cores)

        for mini_pop in split_population[:-1]:
            self._in_queue.put(mini_pop)

        out_list = self.singleCoreFitness(split_population[-1])

        for i in range(self._cores - 1):
            out_list = np.concatenate((out_list, self._out_queue.get()))

        assert(len(out_list) == self._pop_size or len(out_list) == self._pop_size - self._elitism_size)

        return out_list

    def singleCoreFitness(self, population):
        for individual in population:
            self._fitness_function.evaluateSeparateFitnesses(individual)

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

            for h in range(ind_genes.shape[1]):
                maxAngle = angle_limits[h][1]
                minAngle = angle_limits[h][0]

                for i in range(ind_genes.shape[0]):

                    if ind_genes[i, h] > maxAngle:
                        ind_genes[i, h] = maxAngle  # - (ind_genes[i, h] - maxAngle)

                    elif ind_genes[i, h] < minAngle:
                        ind_genes[i, h] = minAngle  # + (minAngle - ind_genes[i,h])

    def sortByFitness(self, population):
        return sorted(population, key=lambda x: x.getFitness(), reverse=True)

    def sharingFunction(self, distance):
        return (1 - distance/self._niche_sigma) if distance < self._niche_sigma else 0

    def selection(self):

        parents_replacing = True
        sorted_pop = self.sortByFitness(self._population)

        # Elitism
        if self._elitism_size > 0:
            self._elite = sorted_pop[:self._elitism_size]

        amount_of_parents = int(self._rate_of_selection * self._pop_size)

        fitness_propotional_parents, pareto_tournament_parents, rank_parents, random_parents = self.percentagesToValues(amount_of_parents, self._selection_method)

        assert(fitness_propotional_parents + pareto_tournament_parents + rank_parents + random_parents == amount_of_parents)

        np.random.shuffle(sorted_pop[:])

        population = sorted_pop

        if fitness_propotional_parents > 0:

            # Probabilities of selection for each individual is calculated
            fitness_values = [ind.getFitness() for ind in population]

            total = sum(fitness_values)
            probabilities = [fitness / total for fitness in fitness_values]

            pop_indexes = range(len(population))
            chosen_indexes = np.random.choice(pop_indexes, size=fitness_propotional_parents, p=probabilities, replace=parents_replacing)
            for index in chosen_indexes:
                self._parents.append(population[index])
            if not parents_replacing:
                population = np.delete(population, chosen_indexes)

            # self._parents = np.random.choice(self._population, size=fitness_propotional_parents, p=probabilities)

        if pareto_tournament_parents > 0:

            # List with individuals indexes
            pop_left = list(range(len(population)))
            pop_fitnesses = np.array([ind.getMultiFitness() for ind in population])
            pop_fitnesses_normalized = (pop_fitnesses - np.mean(pop_fitnesses, axis=0)) / np.std(pop_fitnesses - np.mean(pop_fitnesses, axis=0))

            for i in range(pareto_tournament_parents):
                # Amount of population left
                len_pop_left = len(pop_left)

                # Random selection of two individuals indexes
                ind_1_index = pop_left.pop(np.random.randint(len_pop_left))
                ind_2_index = pop_left.pop(np.random.randint(len_pop_left - 1))

                # Get the actual individuals
                ind_1 = population[ind_1_index]
                ind_2 = population[ind_2_index]

                # Get the normalized fitnesses of each individual
                ind_1_mfitness = pop_fitnesses_normalized[ind_1_index]
                ind_2_mfitness = pop_fitnesses_normalized[ind_2_index]

                # Random sampling
                sampled_group_indexes = np.random.choice(pop_left, size=self._pareto_tournament_size, replace=parents_replacing)
                sampled_group = [pop_fitnesses_normalized[s] for s in sampled_group_indexes]

                ind_1_dominates = np.all([pg.pareto_dominance(ind_1_mfitness, samp) for samp in sampled_group])
                ind_2_dominates = np.all([pg.pareto_dominance(ind_2_mfitness, samp) for samp in sampled_group])

                # Sharing
                if ind_1_dominates == ind_2_dominates:
                    # Calculation of niche size

                    niche_ind_1 = sum(
                        [self.sharingFunction(np.linalg.norm(ind_1_mfitness - ind_mfitness)) for ind_mfitness in
                         np.delete(pop_fitnesses_normalized, ind_1_index, axis=0)])
                    niche_ind_2 = sum(
                        [self.sharingFunction(np.linalg.norm(ind_2_mfitness - ind_mfitness)) for ind_mfitness in
                         np.delete(pop_fitnesses_normalized, ind_2_index, axis=0)])

                    corrected_fit_1 = np.divide(ind_1.getFitness(), niche_ind_1)
                    corrected_fit_2 = np.divide(ind_2.getFitness(), niche_ind_2)

                    if corrected_fit_1 > corrected_fit_2:
                        self._parents.append(ind_1)
                        pop_left.append(ind_2_index)
                    else:
                        self._parents.append(ind_2)
                        pop_left.append(ind_1_index)

                elif ind_1_dominates:
                    self._parents.append(ind_1)
                    pop_left.append(ind_2_index)
                elif ind_2_dominates:
                    self._parents.append(ind_2)
                    pop_left.append(ind_1_index)

            assert (len(population) == self._pop_size - fitness_propotional_parents - pareto_tournament_parents and len(
                self._parents) == fitness_propotional_parents + pareto_tournament_parents)
        # Rank
        if rank_parents > 0:
            probabilities = []
            population = self.sortByFitness(population)

            for i, ind in enumerate(population):
                i += 1 # Starts from one

                # P_i = P_c (1 - P_c) ^ (i - 1), P_n = (1 - P_c) ^ (n - 1)
                probabilities.append((self._rank_probability if i != len(population) else 1) * (1 - self._rank_probability) ** (i - 1))

            pop_indexes = range(len(population))
            chosen_indexes = np.random.choice(pop_indexes, size=rank_parents, p=probabilities, replace=parents_replacing)
            for index in chosen_indexes:
                self._parents.append(population[index])
            if not parents_replacing:
                population = np.delete(population, chosen_indexes)

            # self._parents = np.random.choice(population, size=rank_parents, p=probabilities)

        # Random
        if random_parents > 0:
            pop_indexes = range(len(population))
            chosen_indexes = np.random.choice(pop_indexes, size=random_parents, replace=parents_replacing)
            for index in chosen_indexes:
                self._parents.append(population[index])
            if not parents_replacing:
                population = np.delete(population, chosen_indexes)


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
        while len(self._children) < (self._pop_size - self._elitism_size):
            if i == amount and j == amount - 1:
                i = 0
                j = 0
                coinToss = np.random.rand(amount, amount)

            if coinToss[i,j] < self._pairing_prob and i != j:
                child1, child2 = self.crossover(self._parents[i], self._parents[j])
                self._children.append(child1)
                if len(self._children) < (self._pop_size - self._elitism_size):
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
        self._population = np.concatenate((self._elite, self._children))
        np.random.shuffle(self._population)
        self._parents = []
        self._children = []
        self._elite = []
        self._generation += 1

    def terminationCondition(self):
        best_case_np = np.array(self._best_case)
        generationLimitCondition = self._generation >= self._generation_threshold
        bestIndividualCondition = best_case_np[len(self._best_case) - 1,0] > self._fitness_threshold
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

        fitnesses = [ind.getFitness() for ind in self._population]
        sep_fitnesses = [ind.getMultiFitness() for ind in self._population]

        self._best_case.append(np.concatenate(([np.max(fitnesses)], np.min(sep_fitnesses, axis=0))))
        self._average_case.append(np.concatenate(([np.mean(fitnesses)], np.mean(sep_fitnesses, axis=0))))

    def plotSingleFitness(self, best, average, xlab, ylab, title, choice):

        fig_fitness, ax_fitness = plt.subplots(ncols=1, nrows=1)
        cases = ['mejor caso', 'promedio']

        if choice == 0 or choice >= len(cases):
            ax_fitness.plot(best, label=cases[0])
        if choice == 1 or choice >= len(cases):
            ax_fitness.plot(average, label=cases[1])

        ax_fitness.legend(["Mejor Caso", "Promedio"])
        ax_fitness.set_xlabel(xlab, fontsize=10)
        ax_fitness.set_ylabel(ylab, fontsize=10)
        ax_fitness.set_title(title)

        return fig_fitness

    def graph(self, choice):

        best_case_np = np.array(self._best_case)
        average_case_np = np.array(self._average_case)

        # Fitness
        fig_fitness = self.plotSingleFitness(best_case_np[:, 0], average_case_np[:, 0], "Generación",
                                             "Función de Fitness", "Evolución del algoritmo genético", choice)

        # Distancia
        fig_distancia = self.plotSingleFitness(best_case_np[:, 1], average_case_np[:, 1], "Generación", "Distancia",
                               "Evolución del algoritmo genético", choice)

        # Torque
        fig_torque = self.plotSingleFitness(best_case_np[:, 2], average_case_np[:, 2], "Generación", "Torque",
                               "Evolución del algoritmo genético", choice)

        # plt.show()
        self._graphs_fitness.append([fig_fitness, fig_distancia, fig_torque])

    def graphIndividual(self):
        fig_best_individual, ax_best_individual = plt.subplots(ncols=1, nrows=1)

        for ang in np.transpose(self._best_individual.getGenes()):
            ax_best_individual.plot(ang)

        ax_best_individual.legend([r"$\theta_1$", r"$\theta_2$", r"$\theta_3$", r"$\theta_4$"])
        ax_best_individual.set_title("Mejor individuo")
        ax_best_individual.set_xlabel("Unidad de Tiempo")
        ax_best_individual.set_ylabel("Ángulo [rad]")
        # plt.show()
        self._graphs_individuals.append([fig_best_individual, self._generation])
        self._best_individuals_list.append([self._best_individual.getGenes().tolist(), self._generation])

    def printGenerationData(self):
        t = time.time() - self._start_time
        best_case_np = np.array(self._best_case)
        average_case_np = np.array(self._average_case)

        self._print_module.print("| Generation:                    %4.4d |\n" % (self._generation) +
              "| Best Generation Fitness: %10.8f |\n" % (best_case_np[self._generation - 1,0]) +
              "| Mean Generation Fitness: %10.8f |\n" % (average_case_np[self._generation - 1,0]) +
              "| Best Overall Fitness:    %10.8f |\n" % (max(best_case_np[:,0])) +
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

    def plotParetoFrontier(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        f_distance, f_torque, f_velocity = zip(*[ind.getMultiFitness() for ind in self._population])
        ax.scatter(f_distance, f_torque, f_velocity)
        ax.set_xlim([0, 16])
        ax.set_ylim([0, 8000])
        ax.set_zlim([0, 6])
        fig.show()

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

    def getBestIndividualsList(self):
        return self._best_individuals_list

    def getbestCase(self):
        return self._best_case

    def getAverageCase(self):
        return self._average_case

    def getFitnessGraphs(self):
        return self._graphs_fitness

    def getIndividualsGraphs(self):
        return self._graphs_individuals

    def getTrainingTime(self):
        return self._total_training_time

    def getGeneration(self):
        return self._generation

    def getDefaults(self):
        return self._all_info

    @staticmethod
    def percentagesToValues(total, percentages):
        cum_perc = 0
        cum_value = 0
        values = []
        for perc in percentages:
            cum_perc += perc
            value = int(cum_perc * total) - cum_value
            cum_value += value
            values.append(value)

        return values