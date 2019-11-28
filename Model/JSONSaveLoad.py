import json
import datetime
import itertools
import os
import matplotlib.pyplot as plt


class JSONSaveLoad:

    def __init__(self, GA, save_filename, parameters_variations={}):

        # Genetic Algorithm
        self._GA = GA

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

    def loadParameters(self, repetitions=1, tune_parameters=False, continue_tuning=False):
        runs = []

        default_parameters = self._GA.getAlgorithmInfo()

        keys, values = zip(*self._parameters_variations.items())

        if not tune_parameters:
            for v in itertools.product(*values):

                run = default_parameters.copy()
                run_change = dict(zip(keys, v))
                for key, val in run_change.items():
                    run[key] = val
                runs.append(run)
            final_runs = [r for r in runs for i in range(repetitions)]
        else:
            first_param_run = {key: val[0] for key, val in zip(keys, values)}
            final_runs = []
            repetitions = 2

            if not continue_tuning:
                for _ in range(repetitions):
                    final_runs.append(first_param_run)

            changing_params = {key: val for key, val in zip(keys, values) if len(val) > 1}

            for key, val in changing_params.items():
                for ind_val in val[1:]:
                    new_run = first_param_run.copy()
                    new_run[key] = ind_val
                    for _ in range(repetitions):
                        final_runs.append(new_run)


        self._runs = final_runs

    def getRuns(self):
        return self._runs

    def saveJson(self, GA):

        new_individual = {}

        with open(self._trained_models_dir + "/" + self._save_filename + ".json") as f:
            individuals_json = json.load(f)
            index = individuals_json["Index"]

            fit_graphs, dist_graphs, torque_graphs = GA.getFitnessGraphs()

            fit_graphs.savefig(self._trained_models_dir + "/Graphs/Fitness/fitness_graph_" + str(index))
            dist_graphs.savefig(self._trained_models_dir + "/Graphs/Distance/distance_graph_" + str(index))
            torque_graphs.savefig(self._trained_models_dir + "/Graphs/Torque/torque_graph_" + str(index))

            ind_graphs = GA.getIndividualsGraphs()

            for graph in ind_graphs:
                if not os.path.exists(self._trained_models_dir + f"/Graphs/Individuals/{index}"):
                    os.makedirs(self._trained_models_dir + f"/Graphs/Individuals/{index}")

                graph[0].savefig(self._trained_models_dir + f"/Graphs/Individuals/{index}/best_individual_graph_" + str(index) + "_gen_" + str(graph[1]))

            plt.close('all')

            best_individuals = individuals_json["Best Individuals"]
            new_individual["ID"] = index
            individuals_json["Index"] += 1
            new_individual["Genes"] = GA.getBestIndividualsList()
            new_individual["Info"] = GA.getAlgorithmInfo()
            new_individual["Time of Training"] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            new_individual["Total Training Time"] = GA.getTrainingTime()
            new_individual["Last Generation"] = GA.getGeneration()
            new_individual["Fitness"] = GA.getBestIndividual().getFitness()
            new_individual["Multi Fitness"] = GA.getBestIndividual().getMultiFitness().tolist()
            new_individual["Animate"] = False
            best_individuals.append(new_individual)

        with open(self._trained_models_dir + "/" + self._save_filename + ".json", 'w') as f:
            json.dump(individuals_json, f)

