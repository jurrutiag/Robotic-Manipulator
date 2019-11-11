

class DisplayHandler:

    def __init__(self, GA, print_data=True, generation_for_print=5, console_print=True):
        self._GA = GA
        self._print_data = print_data
        self._generation_for_print = generation_for_print
        self._console_print = console_print

    def updateDisplay(self, terminate=False):

        if terminate:
            # Print last information on the terminal
            if self._print_data:
                self._GA.printGenerationData()
                if self._console_print:
                    self._GA.printGenerationData()
                self._GA.updateInfo(terminate=True)

            self._GA.graph(2)

        else:
            # Information is printed on the terminal
            if self._generation_for_print and self._GA.getGeneration() % self._generation_for_print == 0 and self._print_data:
                if self._console_print:
                    self._GA.printGenerationData()
                self._GA.updateInfo()