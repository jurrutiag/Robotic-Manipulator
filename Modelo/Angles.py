import math

class Angles():

    def __init__(self, theta_1, theta_2, theta_3, theta_4):
        """

        :param theta_1: Angle from X to the arm on XY plane, A1 origin.
        :type theta_1: double
        :param theta_2: Angle from X to L2 on XZ plane, A1 origin.
        :type theta_2: double
        :param theta_3: Angle from X to L3 on XZ plane, A2 origin
        :type theta_3: double
        :param theta_4: Angle from X to L4 on XZ plane, A3 origin.
        :type theta_4: double
        """

        self._thetas = ()

    def getThetas(self):
        return self._thetas
