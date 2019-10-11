import math


class RoboticManipulator:

    def __init__(self, dimensions, mass, angle_limits=([-math.pi / 2, math.pi / 2], [-math.pi / 2, math.pi / 2], [-math.pi / 2, math.pi / 2], [-math.pi / 2, math.pi / 2])):

        self._dimensions = dimensions
        self._mass = mass
        self._angle_limits = angle_limits

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

        L1 = self._dimensions[0]
        L2 = self._dimensions[1]
        L3 = self._dimensions[2]
        L4= self._dimensions[3]

        A3_XYprojection = L2 * math.cos(theta_2)
        A3_height = L1 + L2 * math.sin(theta_2)

        A3_position = [
            A3_XYprojection * math.cos(theta_1),
            A3_XYprojection * math.sin(theta_1),
            A3_height
        ]

        A4_XYprojection = A3_XYprojection + L3 * math.cos(theta_2 + theta_3)
        A4_height = A3_height + L3 * math.sin(theta_2 + theta_3)

        A4_position = [
            A4_XYprojection * math.cos(theta_1),
            A3_XYprojection * math.sin(theta_1),
            A4_height
        ]

        end_effector_XYprojection = A4_XYprojection + L4* math.cos(
            theta_4 + theta_3 + theta_2)
        end_effector_height = A4_height + L4* math.sin(theta_2 + theta_3 + theta_4)

        end_effector_position = [
            end_effector_XYprojection * math.cos(theta_1),
            end_effector_XYprojection * math.sin(theta_1),
            end_effector_height
        ]

        return [0, 0, L1], A3_position, A4_position, end_effector_position

    def getMass(self):
        return self._mass

    def getDimensions(self):
        return self._dimensions

    def getLimits(self):
        return self._angle_limits

if __name__ == "__main__":
    rb = RoboticManipulator((5, 5, 5, 5), (1, 1, 1))



    pos = rb.anglesToPositions(0.7852755591127947, 1.3508413788012184, -1.3461895993542783, -1.3779707726237265)

    r_1 = pos[0]
    r_2 = pos[1]
    r_3 = pos[2]
    r_4 = pos[3]
    print(pos)
    print(r_1, r_2, r_3, r_4)