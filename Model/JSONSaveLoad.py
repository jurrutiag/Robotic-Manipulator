import json
import datetime
import itertools
import os
import matplotlib.pyplot as plt


class JSONSaveLoad:

    def __init__(self, parameters_from_filename, quick_save_filename, save_filename, parameters_variations={}):

        # Filenames
        self._parameters_from_filename = parameters_from_filename
        self._quick_save_filename = quick_save_filename
        self._save_filename = save_filename
        self._trained_models_dir = "Trained Models/" + self._save_filename

        if not os.path.exists(self._trained_models_dir):
            os.makedirs(self._trained_models_dir)
            os.makedirs(self._trained_models_dir + "/Graphs")
            os.makedirs(self._trained_models_dir + "/Graphs/Individuals")
            os.makedirs(self._trained_models_dir + "/Graphs/Torque")
            os.makedirs(self._trained_models_dir + "/Graphs/Fitness")
            os.makedirs(self._trained_models_dir + "/Graphs/Distance")
            with open(self._trained_models_dir + "/" + self._save_filename + ".json", 'w') as f:
                pass
            with open(self._trained_models_dir + "/" + self._save_filename + ".json", 'w') as f:
                json.dump({"Best Individuals": [], "Index": 0}, f)

        # GA
        self._parameters_variations = parameters_variations
        self._parameters = {}

    def loadParameters(self, repetitions=1):
        if self._parameters_from_filename:
            with open(self._parameters_from_filename) as f:
                parameters = json.load(f)
                default_params = parameters["Default"]
                runs = []

                keys, values = zip(*self._parameters_variations.items())

                for v in itertools.product(*values):

                    run = default_params.copy()
                    run_change = dict(zip(keys, v))
                    for key, val in run_change.items():
                        run[key] = val
                    runs.append(run)
                parameters["Runs"] = [r for r in runs for i in range(repetitions)]

            self._parameters = parameters

    def getRuns(self):
        return self._parameters["Runs"]

    def saveJson(self, GA, quick=False):

        json_filename = self._quick_save_filename if quick else self._save_filename

        new_individual = {}

        with open(self._trained_models_dir + "/" + self._save_filename + ".json") as f:
            individuals_json = json.load(f)
            index = individuals_json["Index"]

            fit_graphs, torque_graphs, dist_graphs = GA.getFitnessGraphs()[0]

            fit_graphs.savefig(self._trained_models_dir + "/Graphs/Fitness/fitness_graph_" + str(index))
            torque_graphs.savefig(self._trained_models_dir + "/Graphs/Torque/torque_graph_" + str(index))
            dist_graphs.savefig(self._trained_models_dir + "/Graphs/Distance/distance_graph_" + str(index))

            ind_graphs = GA.getIndividualsGraphs()

            for graph in ind_graphs:
                graph[0].savefig(self._trained_models_dir + "/Graphs/Individuals/best_individual_graph_" + str(index) + "_gen_" + str(graph[1]))

            plt.close('all')

            best_individuals = individuals_json["Best Individuals"]
            new_individual["ID"] = index
            individuals_json["Index"] += 1
            new_individual["Genes"] = GA.getBestIndividual().getGenes().tolist()
            new_individual["Info"] = GA.getAlgorithmInfo()
            new_individual["Time of Training"] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            new_individual["Total Training Time"] = GA.getTrainingTime()
            new_individual["Last Generation"] = GA.getGeneration()
            new_individual["Fitness"] = GA.getBestIndividual().getFitness()
            new_individual["Animate"] = False
            best_individuals.append(new_individual)

        with open(self._trained_models_dir + "/" + self._save_filename + ".json", 'w') as f:
            json.dump(individuals_json, f)

