import json
import datetime
import itertools
import os
import matplotlib.pyplot as plt


class JSONSaveLoad:

    json_config_folder = "JSON files"
    prev_parameters_dir = json_config_folder + "/runs_parameters.json"
    quick_models_dir = json_config_folder + "/quick_models.json"

    def __init__(self, save_filename, parameters_variations={}):

        # Filenames
        self._save_filename = save_filename
        self._trained_models_dir = "Trained Models/" + self._save_filename

        if not os.path.exists(self._trained_models_dir):
            os.makedirs(self._trained_models_dir)
            os.makedirs(self._trained_models_dir + "/Renders")
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
        self._runs = None

    def loadParameters(self, repetitions=1):
        if self.prev_parameters_dir:
            runs = []

            default_parameters = self.loadDefaults()

            keys, values = zip(*self._parameters_variations.items())

            for v in itertools.product(*values):

                run = default_parameters.copy()
                run_change = dict(zip(keys, v))
                for key, val in run_change.items():
                    run[key] = val
                runs.append(run)
            final_runs = [r for r in runs for i in range(repetitions)]

            self._runs = final_runs

    def getRuns(self):
        return self._runs

    def saveJson(self, GA, quick=False):

        json_filename = self.quick_models_dir if quick else self._save_filename

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
                if not os.path.exists(self._trained_models_dir + f"/Graphs/Individuals/{index}"):
                    os.makedirs(self._trained_models_dir + f"/Graphs/Individuals/{index}")
                graph[0].savefig(self._trained_models_dir + f"/Graphs/Individuals/{index}/best_individual_graph_" + str(index) + "_gen_" + str(graph[1]))

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

    @staticmethod
    def loadDefaults():
        with open(JSONSaveLoad.prev_parameters_dir, 'rb') as f:
            parameters = json.load(f)
            default_parameters = parameters["Default"]

        return default_parameters
