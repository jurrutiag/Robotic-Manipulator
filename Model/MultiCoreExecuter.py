import json
from GeneticAlgorithm import GeneticAlgorithm
import multiprocessing
import time
import os
from PrintModule import PrintModule


class MultiCoreExecuter:

    def __init__(self, runs, manipulator, json_handler, cores=4):

        self._manipulator = manipulator
        self._runs = runs
        self._json_handler = json_handler
        self._cores = cores

    def run(self):
        print_module = PrintModule()
        print_module.initialize()
        print_module.clear()

        t0 = time.time()
        if self._cores == 1:
            for i, run in enumerate(self._runs):
                print_module.print(f"Run number {i + 1}, with parameters: " + json.dumps(run), position="Current Information", color="red")
                GA = GeneticAlgorithm(self._manipulator, print_module=print_module, **run)
                GA.runAlgorithm()

                self._json_handler.saveJson(GA)

        else:
            print_module.setCores(self._cores)

            print_module.assignLock(multiprocessing.Lock())
            runs_queue = multiprocessing.Queue()

            for run in self._runs:
                runs_queue.put(run)

            processes = []

            print_module.print(f"Running normal algorithm, multicore models with {self._cores} cores...", position="Current Information")

            for i in range(self._cores):
                p = multiprocessing.Process(target=self.notBatchesExecuter, args=(runs_queue, print_module, i))
                processes.append(p)
                p.start()

            for p in processes:
                p.join()

        print_module.print(f"Finished. Total time: {time.time() - t0}", position="Final Information")

    def GAExecuter(self, GA):
        GA.runAlgorithm()
        self._json_handler.saveJson(GA)

    def notBatchesExecuter(self, q_runs, print_module, n_proc):
        i = 1
        print_module.initialize()
        print_module.setProcess(n_proc)
        while not q_runs.empty():
            run = q_runs.get()
            print_module.print(f"Running process {os.getpid()} on run {i} of this process.", color='red', position="Quick Information")
            GA = GeneticAlgorithm(self._manipulator, print_module=print_module, **run)
            self.GAExecuter(GA)
            i += 1