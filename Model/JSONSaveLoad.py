import json
import datetime
import itertools
import os


class JSONSaveLoad:

    def __init__(self, parameters_from_filename, parameters_visualization_filename, quick_save_filename, save_filename, parameters_variations={}):

        # Filenames
        self._parameters_from_filename = parameters_from_filename
        self._parameters_visualization_filename = parameters_visualization_filename
        self._quick_save_filename = quick_save_filename
        self._save_filename = save_filename
        self._trained_models_dir = "Trained Models/" + self._save_filename

        if not os.path.exists(self._trained_models_dir):
            os.makedirs(self._trained_models_dir)
            os.makedirs(self._trained_models_dir + "/Graphs")
            with open(self._trained_models_dir + "/" + self._save_filename + ".json", 'w') as f:
                pass
            with open(self._trained_models_dir + "/" + self._save_filename + ".json", 'w') as f:
                json.dump({"Best Individuals": [], "Index": 0}, f)

        # GA
        self._parameters_variations = parameters_variations
        self._parameters = {}

    def loadParameters(self):
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
                parameters["Runs"] = runs

            self._parameters = parameters

    def parametersToJSON(self):
        with open(self._parameters_visualization_filename, 'w') as f:
            json.dump(self._parameters, f)

    def getRuns(self):
        return self._parameters["Runs"]

    def saveJson(self, GA, quick=False):

        json_filename = self._quick_save_filename if quick else self._save_filename

        new_individual = {}



        with open(self._trained_models_dir + "/" + self._save_filename + ".json") as f:
            individuals_json = json.load(f)
            index = individuals_json["Index"]

            graphs = GA.getGraphs()
            graphs[0].savefig(self._trained_models_dir + "/Graphs/fitness_graph_" + str(index))
            graphs[1].savefig(self._trained_models_dir + "/Graphs/best_individual_graph_" + str(index))

            best_individuals = individuals_json["Best Individuals"]
            new_individual["ID"] = index
            individuals_json["Index"] += 1
            new_individual["Genes"] = GA.getBestIndividual().getGenes().tolist()
            new_individual["Info"] = GA.getAlgorithmInfo()
            new_individual["Time of Training"] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            new_individual["Fitness"] = GA.getBestIndividual().getFitness()
            new_individual["Animate"] = False
            best_individuals.append(new_individual)

        with open(self._trained_models_dir + "/" + self._save_filename + ".json", 'w') as f:
            json.dump(individuals_json, f)

