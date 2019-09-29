

class GA:

    def __init__(self, argg1, a=1, b=1, c=1):
        print(argg1, a, b, c)


def wrapper(func, arg1, kwargs):
    return func(arg1, **kwargs)


if __name__ == "__main__":
    wrapper(GA, 10, {'a': 5, 'c': 6, 'b': 7})


def geneticWrapper(func, desired_position, manipulator, args):
    return func(desired_position, manipulator, *args)


from_file_filename = "all_runs.json"
from_file = True
    if from_file:
        with open(from_file_filename) as f:
            file = json.load(f)
            runs = file["Runs"]
            for run in runs:
                GA = geneticWrapper(GeneticAlgorithm, desired_position, manipulator, run)
                saveJson(json_filename, GA)
