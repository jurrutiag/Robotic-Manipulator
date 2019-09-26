import math


class RoboticManipulator:

    def __init__(self, dimensions, mass):

        self._L1 = dimensions[0]
        self._L2 = dimensions[1]
        self._L3 = dimensions[2]
        self._L4 = dimensions[3]

        self._mass = mass

    def anglesToPositions(self, theta_1, theta_2, theta_3, theta_4):
        """

        :param theta_1: Angle in radians from X to the arm on XY plane, A1 origin
        :type theta_1: double
        :param theta_2: Angle in radians from X to L2 on XZ plane, A2 origin
        :type theta_2: double
        :param theta_3: Angle in radians from X to L3 on XZ plane, A3 origin
        :type theta_3: double
        :param theta_4: Angle in radians from X to L4 on XZ plane, A4 origin
        :type theta_4: double
        :return: None
        :rtype: None
        """

        A3_XYprojection = self._L2 * math.cos(theta_2)
        A3_height = self._L2 * math.sin(theta_2)

        A3_position = [
            A3_XYprojection * math.cos(theta_1),
            A3_XYprojection * math.sin(theta_1),
            A3_height
        ]

        A4_XYprojection = A3_XYprojection + self._L3 * math.cos(theta_2 + theta_3)
        A4_height = A3_height + self._L3 * math.sin(theta_2 + theta_3)

        A4_position = [
            A4_XYprojection * math.cos(theta_1),
            A3_XYprojection * math.sin(theta_1),
            A4_height
        ]

        end_effector_XYprojection = A4_XYprojection + self._L4 * math.cos(
            theta_4 + theta_3 + theta_2)
        end_effector_height = A4_height + self._L4 * math.sin(theta_2 + theta_3 + theta_4)

        end_effector_position = [
            end_effector_XYprojection * math.cos(theta_1),
            end_effector_XYprojection * math.sin(theta_1),
            end_effector_height
        ]

        return A3_position, A4_position, end_effector_position

    def getMass(self):
        return self._mass