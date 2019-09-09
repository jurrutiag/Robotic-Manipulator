import math


class RoboticManipulator:

    def __init__(self, L1, L2, L3, L4, position=[0, 0, 0]):

        self._L1 = L1
        self._L2 = L2
        self._L3 = L3
        self._L4 = L4

        self._position = position

    def anglesToPosition(self, theta_1, theta_2, theta_3, theta_4):
        """

        :param theta_1: Angle from X to the arm on XY plane, A1 origin
        :type theta_1: double
        :param theta_2: Angle from X to L2 on XZ plane, A1 origin
        :type theta_2: double
        :param theta_3: Angle from X to L3 on XZ plane, A2 origin
        :type theta_3: double
        :param theta_4: Angle from X to L4 on XZ plane, A3 origin
        :type theta_4: double
        :return: None
        :rtype: None
        """
        theta_1 = math.radians(theta_1)
        theta_2 = math.radians(theta_2)
        theta_3 = math.radians(theta_3)
        theta_4 = math.radians(theta_4)

        XYprojection = self._L2 * math.cos(theta_2) + self._L3 * math.cos(theta_3 + theta_2) + self._L4 * math.cos(theta_4 + theta_3 + theta_2)

        self._position[0] = XYprojection * math.cos(theta_1)
        self._position[1] = XYprojection * math.sin(theta_1)
        self._position[2] = self._L1 + self._L2 * math.sin(theta_2) + self._L3 * math.sin(theta_3 + theta_2) + self._L4 * math.sin(theta_4 + theta_3 + theta_2)

    def getPosition(self):
        return self._position
