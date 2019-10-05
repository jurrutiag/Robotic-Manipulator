import colorama


class PrintModule:

    def __init__(self, position):
        self._position = position


    def initialize(self):
        colorama.init()

    def clear(self):
        PrintModule.clear()

    def print(self, string, position=None, color='white'):
        if position is None:
            position = self._position

        if color == 'white':
            color = colorama.Fore.WHITE
        elif color == 'red':
            color = colorama.Fore.RED

        PrintModule.move_cursor(0, position)
        print(color + string + colorama.Fore.RESET)

    @staticmethod
    def move_cursor(x, y):
        print("\x1b[{};{}H".format(y + 1, x + 1))

    @staticmethod
    def clear():
        print("\x1b[2J")

    def getPosition(self):
        return self._position

    def setPosition(self, position):
        self._position = position
