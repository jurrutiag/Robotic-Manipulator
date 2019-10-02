import json
from GeneticAlgorithm import GeneticAlgorithm
import multiprocessing
import time


class MultiCoreExecuter:

    def __init__(self, runs, manipulator, json_handler, cores=4):

        self._manipulator = manipulator
        self._runs = runs
        self._json_handler = json_handler
        self._cores = cores

    def run(self):
        t0 = time.time()
        if self._cores == 1:
            for i, run in enumerate(self._runs):
                print(f"Run number {i + 1}, with parameters: " + json.dumps(run))

                GA = MultiCoreExecuter.geneticWrapper(GeneticAlgorithm, self._manipulator, run)
                GA.runAlgorithm()

                self._json_handler.saveJson(GA)
        else:

            runs_batches = []
            current_batch = []

            for i, run in enumerate(self._runs):
                if i % self._cores == 0:
                    if i != 0:
                        runs_batches.append(current_batch.copy())
                    current_batch = [run]
                else:
                    current_batch.append(run)
            else:
                runs_batches.append(current_batch.copy())

            batches_length = len(runs_batches)

            for i, run_batch in enumerate(runs_batches):
                n_processes = min(self._cores, len(run_batch))

                print(f"Model Batch number {i + 1} of {batches_length}.")

                GAs = [MultiCoreExecuter.geneticWrapper(GeneticAlgorithm, self._manipulator, run_batch[i]) for i in range(n_processes)]
                new_GAs = []

                processes = []
                qs = []

                for i in range(n_processes):
                    q = multiprocessing.Queue()
                    qs.append(q)
                    p = multiprocessing.Process(target=MultiCoreExecuter.GAExecuter, args=(q, GAs[i]))
                    processes.append(p)
                    p.start()

                for i in range(n_processes):
                    new_GAs.append(qs[i].get())

                for p in processes:
                    p.join()

                for ga in new_GAs:
                    self._json_handler.saveJson(ga)

        print(f"Total time: {time.time() - t0}")

    @staticmethod
    def geneticWrapper(func, manipulator, kwargs):
        return func(manipulator, **kwargs)

    @staticmethod
    def GAExecuter(q, GA):
        GA.runAlgorithm()
        q.put(GA)
