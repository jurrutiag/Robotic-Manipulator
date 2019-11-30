import math
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import Individual
import FitnessFunction
import time
from PrintModule import PrintModule
import multiprocessing
import pygmo as pg
import itertools
from DisplayHandler import DisplayHandler
import sys
from MultiCoreExecuter import MultiCoreExecuter
from mpl_toolkits.mplot3d import Axes3D  # No borrar, es necesario


class GeneticAlgorithm:

    DEFAULT_POSITION = [5, 5, 5]

    def __init__(self, manipulator,
                 desired_position=DEFAULT_POSITION,
                 print_module=None,
                 cores=1,
                 pop_size=150,
                 cross_individual_prob=0.4,
                 mut_individual_prob=0.5,
                 cross_joint_prob=0.75,
                 mut_joint_prob=0.25,
                 pairing_prob=0.5,
                 sampling_points=20,
                 torques_ponderations=(1, 1, 1, 1),
                 generation_threshold=100,
                 fitness_threshold=1,
                 progress_threshold=1,
                 generations_progress_threshold=50,
                 torques_error_ponderation=0.1,
                 distance_error_ponderation=1,
                 velocity_error_ponderation=0,
                 rate_of_selection=0.4,
                 elitism_size=15,
                 selection_method=[0, 0.2, 0.6, 0.2],
                 rank_probability=0.4,
                 pareto_tournament_size=5,
                 niche_sigma=0.2,
                 generation_for_print=10,
                 print_data=True,
                 exponential_initialization=False,
                 total_time=5,
                 individuals_to_display=5,
                 model_process_and_id=(1, 1)):

        # Algorithm info for save

        self._all_info = locals().copy()
        del self._all_info["print_module"]
        del self._all_info["manipulator"]
        del self._all_info["self"]
        del self._all_info["model_process_and_id"]

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

        #self._individuals_to_display = (np.linspace(1, generation_threshold, individuals_to_display, dtype=int) if individuals_to_display >= 2 else [pop_size])
        self._individuals_to_display = individuals_to_display

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
        self._start_time = 0
        self._total_training_time = None

        # Final Results

        self._best_individual = None
        self._graphs_fitness = []
        self._graphs_individuals = []
        self._quick_individuals_graphs = []
        self._best_individuals_candidates = []
        self._best_individuals_list = []
        self._last_pareto_frontier = None

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

        # Information Display

        self._display_handler = DisplayHandler(self,
                                               print_data=print_data,
                                               generation_for_print=generation_for_print,
                                               console_print=True)
        self._info_queue = None

        self._model_process_and_id = model_process_and_id

    def runAlgorithm(self):

        self._start_time = time.time()

        # First generation gets created
        self.initialization()

        # First generation fitness
        self._population = self.evaluateFitness(self._population)
        self.getBestAndAverage()

        self.angleCorrection()
        self.findBestIndividual()

        while True:

            # Algorithm interruption
            if MultiCoreExecuter.INTERRUPTING_QUEUE is not None and not MultiCoreExecuter.INTERRUPTING_QUEUE.empty():
                interrupting_info = MultiCoreExecuter.INTERRUPTING_QUEUE.get()
                if interrupting_info == "exit":
                    print("Interrupted...")
                    self.buryProcesses()
                    sys.exit(0)

            # if self._generation in self._individuals_to_display:
            #     self.graphIndividual()
            #     self._best_individuals_list.append([self._best_individual.getGenes().tolist(), self._generation])
            if self._generation % 2 == 0 or self._generation in [1, self._generation_threshold]:
                self._best_individuals_candidates.append([self._best_individual.getGenes().tolist(), self._generation, self._best_individual.getFitness()])

            self._display_handler.updateDisplay()

            # Check for termination condition
            if self.terminationCondition():
                self._total_training_time = time.time() - self._start_time
                self.findBestIndividual()

                # Get individuals to plot
                max_fitness = self.getBestIndividual().getFitness()
                each_fitness_percentage = 1 / (self._individuals_to_display + 1)
                current_percentage = each_fitness_percentage
                for ind in self._best_individuals_candidates:
                    if ind[1] in [1, self._generation_threshold]:
                        self._best_individuals_list.append(ind)
                        self.graphIndividual(ind[0], ind[1])
                        continue

                    if ind[2] >= max_fitness * current_percentage:
                        self._best_individuals_list.append(ind)
                        self.graphIndividual(ind[0], ind[1])
                        current_percentage += each_fitness_percentage
                print(self._best_individuals_list)


                self._display_handler.updateDisplay(terminate=True)

                # Bury the processes
                if self._cores > 1:
                    self.buryProcesses()

                return

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
            assert (len(
                self._parents) == fitness_propotional_parents)

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

            assert (len(
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
            assert (len(
                self._parents) == fitness_propotional_parents + pareto_tournament_parents + rank_parents)

        # Random
        if random_parents > 0:
            pop_indexes = range(len(population))
            chosen_indexes = np.random.choice(pop_indexes, size=random_parents, replace=parents_replacing)
            for index in chosen_indexes:
                self._parents.append(population[index])
            if not parents_replacing:
                population = np.delete(population, chosen_indexes)

            assert (len(
                self._parents) == fitness_propotional_parents + pareto_tournament_parents + rank_parents + random_parents)

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
        # parents_amount = len(self._parents)
        # parents_combinations = itertools.product(range(parents_amount), range(parents_amount))
        # parents_combinations_no_rep = [(p1, p2) for p1, p2 in parents_combinations if p1 > p2]
        #
        # pairs_of_parents_indexes = np.random.choice(len(parents_combinations_no_rep), size=int(np.ceil((self._pop_size - self._elitism_size) / 2)))
        #
        # for pair_index in pairs_of_parents_indexes:
        #     pair = parents_combinations_no_rep[pair_index]
        #     child1, child2 = self.crossover(self._parents[pair[0]], self._parents[pair[1]])
        #     self._children.append(child1)
        #     if len(self._children) < (self._pop_size - self._elitism_size):
        #         self._children.append(child2)

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

        self._best_case.append(np.concatenate(([np.max(fitnesses)], sep_fitnesses[np.argmax(fitnesses)])))
        self._average_case.append(np.concatenate(([np.mean(fitnesses)], np.mean(sep_fitnesses, axis=0))))

    def plotSingleFitness(self, best, average, xlab, ylab, title, choice, logscale):

        fig_fitness, ax_fitness = plt.subplots(ncols=1, nrows=1)
        cases = ['mejor caso', 'promedio']

        if choice == 0 or choice >= len(cases):
            ax_fitness.plot(best, label=cases[0])
        if choice == 1 or choice >= len(cases):
            ax_fitness.plot(average, label=cases[1])

        if logscale:
            ax_fitness.set_yscale("log")

        ax_fitness.legend(["Mejor Caso", "Promedio"])
        ax_fitness.set_xlabel(xlab, fontsize=10)
        ax_fitness.set_ylabel(ylab, fontsize=10)
        ax_fitness.set_title(title)

        return fig_fitness

    def graph(self, choice):
        for fit_fig in self._graphs_fitness:
            plt.close(fit_fig)

        best_case_np = np.array(self._best_case)
        average_case_np = np.array(self._average_case)

        # Fitness
        fig_fitness = self.plotSingleFitness(best_case_np[:, 0], average_case_np[:, 0], "Generación",
                                             "Función de Fitness", "Evolución del algoritmo genético", choice, False)

        # Distancia
        fig_distancia = self.plotSingleFitness(best_case_np[:, 1], average_case_np[:, 1], "Generación", "Distancia",
                               "Evolución del algoritmo genético", choice, True)

        # Torque
        fig_torque = self.plotSingleFitness(best_case_np[:, 2], average_case_np[:, 2], "Generación", "Torque",
                               "Evolución del algoritmo genético", choice, True)

        # plt.show()
        self._graphs_fitness = [fig_fitness, fig_distancia, fig_torque]

        # self._graphs_fitness.append([fig_fitness, fig_distancia, fig_torque])

    def graphIndividual(self, individual_genes, generation, quick_graph=False):

        for ind_fig in self._quick_individuals_graphs:
            plt.close(ind_fig)

        fig_best_individual, ax_best_individual = plt.subplots(ncols=1, nrows=1)

        for ang in np.transpose(individual_genes):
            ax_best_individual.plot(ang)

        ax_best_individual.legend([r"$\theta_1$", r"$\theta_2$", r"$\theta_3$", r"$\theta_4$"])
        ax_best_individual.set_title("Mejor individuo")
        ax_best_individual.set_xlabel("Unidad de Tiempo")
        ax_best_individual.set_ylabel("Ángulo [rad]")

        # plt.show()

        self._quick_individuals_graphs = [fig_best_individual, generation]
        if not quick_graph:
            self._graphs_individuals.append([fig_best_individual, generation])


    def printGenerationData(self):
        t = time.time() - self._start_time
        best_case_np = np.array(self._best_case)
        average_case_np = np.array(self._average_case)

        self._print_module.print("| Generation:                    %4.4d |\n" % (self._generation) +
              "| Best Generation Fitness: %10.8f |\n" % (best_case_np[self._generation - 1, 0]) +
              "| Best Generation Dist. :  %10.8f |\n" % (best_case_np[self._generation - 1, 1]) +
              "| Best Generation Torque:  %10f |\n" % (best_case_np[self._generation - 1, 2]) +
              "| Best Generation Vel.:    %10.8f |\n" % (best_case_np[self._generation - 1, 3]) +
              "| Mean Generation Fitness: %10.8f |\n" % (average_case_np[self._generation - 1, 0]) +
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

    def paretoFrontierIndividuals(self, population):
        multi_fitnesses = [list(ind.getMultiFitness()) for ind in population]
        dominants = []
        for multi_fitness in multi_fitnesses:
            if not np.any([pg.pareto_dominance(others_fitness, multi_fitness) for others_fitness in multi_fitnesses if others_fitness != multi_fitness]):
                dominants.append(multi_fitness)

        return dominants

    def plotParetoFrontier(self):
        if self._last_pareto_frontier is not None:
            plt.close(self._last_pareto_frontier)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        f_distance, f_torque, f_velocity = zip(*[ind.getMultiFitness() for ind in self._population])
        dominants_x, dominants_y, dominants_z = zip(*self.paretoFrontierIndividuals(self._population))
        ax.scatter(f_distance, f_torque, f_velocity, zorder=-1)
        ax.scatter(dominants_x, dominants_y, dominants_z, color='red', zorder=1, alpha=1)
        xlim = [0, 16]
        ylim = [0, 8000]
        zlim = [0, 6]

        ax.set_xlim(xlim)
        ax.set_xticks([0, 4, 8, 12, 16])
        ax.set_ylim(ylim)
        ax.set_yticks([0, 2000, 4000, 6000, 8000])
        ax.set_zlim(zlim)
        ax.set_zticks([0, 2, 4, 6])

        ax.set_title("Multiple Fitnesses for Individuals")
        ax.set_xlabel("Distance")
        ax.set_ylabel("Torque")
        ax.set_zlabel("Velocity")

        self._last_pareto_frontier = fig

    def buryProcesses(self):

        for i in range(self._cores - 1):
            self._in_queue.put("finish")

        for p in self._processes:
            p.join()

    def setInfoQueue(self, queue):
        self._info_queue = queue
        self._info_queue.put({"Defaults": self._all_info, "model": self._model_process_and_id})
        self._console_print = False

    def updateInfo(self, terminate=False):
        if self._info_queue is not None:
            self.graph(2)
            self.graphIndividual(self._best_individual.getGenes(), self._generation, quick_graph=True)
            self.plotParetoFrontier()

            info = {
                "model": self._model_process_and_id,
                "generation": self._generation,
                "best_multi_fitness": self._best_case[-1][1:],
                "best_fitness": self._best_case[-1][0],
                "mean_fitness": self._average_case[-1][0],
                "time_elapsed": time.time() - self._start_time,
                "fitness_graph": self._graphs_fitness[0],
                "individual_graph": self._quick_individuals_graphs[0],
                "pareto_graph": self._last_pareto_frontier,
                "terminate": terminate
            }
            self._info_queue.put(info)

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

    @staticmethod
    def getDefaults():
        GA = GeneticAlgorithm(None, GeneticAlgorithm.DEFAULT_POSITION)
        return GA.getAlgorithmInfo()
