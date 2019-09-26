import RoboticManipulator

class GeneticAlgorithm:

    def __init__(self, pop_size=100, cross_individual_prob=0.5, mut_individual_prob=0.1, cross_joint_prob=0.5, mut_joint_prob=0.5, sampling_points=50, manipulator_dimensions=[1, 1, 1, 1]):

        # Algorithm parameters

        self._pop_size = pop_size
        self._cross_individual_prob = cross_individual_prob
        self._mut_individual_prob = mut_individual_prob
        self._cross_joint_prob = cross_joint_prob
        self._mut_joint_prob = mut_joint_prob
        self._sampling_points = sampling_points # N_k

        # Algorithm variables

        self._population = []

        # Manipulator

        self._manipulator = RoboticManipulator(manipulator_dimensions)