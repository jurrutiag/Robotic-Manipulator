import math

class RoboticManipulator():
    def __init__(self, L1, L2, L3, L4, position=[0, 0, 0]):

        self._L1 = L1
        self._L2 = L2
        self._L3 = L3
        self._L4 = L4

        self._position = position

    def anglesToPosition(self, theta_1, theta_2, theta_3, theta_4):
        """

        :param angles: Angles for the arm to move to.
        :type angles: Angles

        """

        XYprojection = self._L2 * math.cos(theta_1) + self._L3 + self._L4 * math.cos(theta_4)

        self._position[0] = XYprojection * math.cos(theta_1)
        self._position[1] = XYprojection * math.sin(theta_1)
        self._position[2] = self._L1 + self._L2 * math.sin(theta_2) + self._L3 * math.sin(theta_3) + self._L4 * math.sin(theta_4)

    def getPosition(self):
        return self._position