import json
from Model import GeneticAlgorithm
import multiprocessing
import time
import os
from Model.PrintModule import PrintModule
from itertools import repeat



class MultiCoreExecuter:

    def __init__(self, runs, manipulator, json_handler, processes=4, dedicated_screen=False, model_name='json_test'):

        self._manipulator = manipulator
        self._runs = runs
        self._json_handler = json_handler
        self._processes = processes

        self._dedicated_screen = dedicated_screen
        self._event = None

        if dedicated_screen:
            self._manager = multiprocessing.Manager()
            self._event = self._manager.Event()
            self._info_queue = self._manager.Queue()
            from InfoDisplay.InformationWindow import runInfoDisplay
            self._info_process = multiprocessing.Process(target=runInfoDisplay, args=(self._info_queue, model_name, self._event, len(runs)))
            self._info_process.start()

    def run(self):
        print_module = PrintModule()
        print_module.initialize()
        print_module.clear()

        t0 = time.time()
        if self._processes == 1:
            for i, run in enumerate(self._runs):
                print_module.print(f"Run number {i + 1}, with parameters: " + json.dumps(run), position="Current Information", color="red")
                GA = GeneticAlgorithm.GeneticAlgorithm(self._manipulator, print_module=print_module, model_process_and_id=(1, i + 1), interrupt_event=self._event, **run)
                self.GAExecuter(GA)

        else:
            print_module.setCores(self._processes)

            print_module.assignLock(multiprocessing.Lock())

            print_module.print(f"Running normal algorithm, multicore models with {self._processes} cores...", position="Current Information")

            p = multiprocessing.Pool(self._processes)
            multiCoreWorker = MultiCoreWorker(self._manipulator, self._json_handler, self._dedicated_screen, self._info_queue)

            for i, run in enumerate(self._runs):
                run["index"] = i + 1
                p.apply_async(multiCoreWorker.work, (run, self._event))
            p.close()

            self._event.wait()
            p.terminate()
            p.join()

        print_module.print(f"Finished. Total time: {time.time() - t0}", position="Final Information")
        if self._dedicated_screen:
            self._info_process.join()

    def GAExecuter(self, GA):
        if self._dedicated_screen:
            GA.setInfoQueue(self._info_queue)
        GA.runAlgorithm()
        self._json_handler.saveJson(GA)


class MultiCoreWorker:
    def __init__(self, manipulator, json_handler, dedicated_screen, info_queue):
        self._manipulator = manipulator
        self._json_handler = json_handler
        self._dedicated_screen = dedicated_screen
        self._info_queue = info_queue

    def work(self, run, event):
        index = run["index"]
        del run["index"]
        GA = GeneticAlgorithm.GeneticAlgorithm(self._manipulator, print_module=None,
                                               model_process_and_id=(os.getpid(), index), interrupt_event=event, **run)
        if self._dedicated_screen:
            GA.setInfoQueue(self._info_queue)

        GA.runAlgorithm()

        self._json_handler.saveJson(GA)