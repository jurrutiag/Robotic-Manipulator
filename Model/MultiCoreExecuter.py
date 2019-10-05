import json
from GeneticAlgorithm import GeneticAlgorithm
import multiprocessing
import time
import os
from PrintModule import PrintModule


class MultiCoreExecuter:

    def __init__(self, runs, manipulator, json_handler, cores=4, by_batches=False):

        self._manipulator = manipulator
        self._runs = runs
        self._json_handler = json_handler
        self._cores = cores
        self._by_batches = by_batches

    def run(self):
        print_module = PrintModule(0)
        print_module.initialize()
        print_module.clear()

        t0 = time.time()
        if self._cores == 1:
            for i, run in enumerate(self._runs):
                print(f"Run number {i + 1}, with parameters: " + json.dumps(run))

                GA = MultiCoreExecuter.geneticWrapper(GeneticAlgorithm, self._manipulator, run)
                GA.runAlgorithm()

                self._json_handler.saveJson(GA)
        elif self._by_batches:

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

                if i % 16 == 0 and i != 0:
                    print("Sleeping for 20 secs.")
                    time.sleep(20)
                    print("Continuing execution.")

                n_processes = min(self._cores, len(run_batch))

                print(f"Model Batch number {i + 1} of {batches_length}.")

                GAs = [MultiCoreExecuter.geneticWrapper(GeneticAlgorithm, self._manipulator, run_batch[i]) for i in range(n_processes)]
                new_GAs = []

                processes = []

                for i in range(n_processes):
                    p = multiprocessing.Process(target=self.GAExecuter, args=(GAs[i],))
                    processes.append(p)
                    p.start()

                for p in processes:
                    p.join()

                for ga in new_GAs:
                    self._json_handler.saveJson(ga)

        else:

            runs_queue = multiprocessing.Queue()

            for run in self._runs:
                runs_queue.put(run)

            processes = []


            for i in range(self._cores):
                terminal_position = 8 * i
                print_module.setPosition(terminal_position)
                p = multiprocessing.Process(target=self.notBatchesExecuter, args=(runs_queue, print_module))
                processes.append(p)
                p.start()

            print_module.setPosition(8 * self._cores)

            for p in processes:
                p.join()

        print_module.print(f"Total time: {time.time() - t0}")

    @staticmethod
    def geneticWrapper(func, manipulator, kwargs):
        return func(manipulator, **kwargs)

    def GAExecuter(self, GA):
        GA.runAlgorithm()
        self._json_handler.saveJson(GA)

    def notBatchesExecuter(self, q_runs, print_module):
        i = 1
        while not q_runs.empty():
            print_module.initialize()
            run = q_runs.get()
            print_module.print(f"Running process {os.getpid()} on run {i} of this process.", color='red')
            GA = GeneticAlgorithm(print_module, self._manipulator, **run)
            self.GAExecuter(GA)
            i += 1