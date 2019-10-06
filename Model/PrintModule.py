import colorama


class PrintModule:

    def __init__(self):
        self._positions = {"Current Information": (0, 5), "Quick Information": (6, 1), "Current Training": (7, 6), "Final Information": (14, 1)}
        self._process = 1
        self._lock = None

    def initialize(self):
        colorama.init()

    def print(self, string, position=None, color='white'):
        if self._lock is not None:
            self._lock.acquire()

        self.clearLines(position)
        position = self._positions[position]

        if color == 'white':
            color = colorama.Fore.WHITE
        elif color == 'red':
            color = colorama.Fore.RED

        PrintModule.move_cursor(0, position[0])
        print(color + string + colorama.Fore.RESET, end="")

        if self._lock is not None:
            self._lock.release()

    @staticmethod
    def move_cursor(x, y):
        print("\x1b[{};{}H".format(y + 1, x + 1))

    def clearLines(self, info_type):
        if self._positions[info_type][1] == -1:
            return
        for i in range(self._positions[info_type][1]):
            PrintModule.move_cursor(0, self._positions[info_type][0] + i)
            print("\033[K")

    @staticmethod
    def clear():
        print("\x1b[2J")

    def getPositions(self):
        return self._positions

    def setPositions(self, positions):
        self._positions = positions

    def addPosition(self, position_name, position, length):
        self._positions[position_name] = (position, length)

    def assignLock(self, lock):
        self._lock = lock

    def setProcess(self, n_proc):
        self._positions["Quick Information"] = (7 * n_proc + 7, 6)
        self._positions["Current Training"] = (7 * n_proc + 8, 6)

    def setCores(self, cores):
        self._positions["Final Information"] = (7 * (cores - 1) + 8 + 7, 1)
