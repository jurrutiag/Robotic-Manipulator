import RoboticManipulator
import numpy as np
import math

class FitnessFunction:

    def __init__(self, manipulator):
        self._manipulator = manipulator


    def evaluateFitness(self, individual):
        pass

    #retorna posiciones cartesionas de las tres masas móviles del brazo
    def getPositions(self, individual):

        # a2x, a2y, a2z, a3x, a3y, a3z, a4x, a4y, a4z

        positions = []

        for ang in individual.getGenes():

            theta_1, theta_2, theta_3, theta_4 = ang

            position_2, position_3, position_4 = self._manipulator.anglesToPositions(theta_1, theta_2, theta_3, theta_4)

            thisPosition= np.hstack([position_2, position_3, position_4])

            positions.append(thisPosition)

        return np.array(positions)


    def getAngularAccelerations(self, individual):

        angles = individual.getGenes()

        n_samples = len(angles)

        accelerations = []
        for i, angle in enumerate(angles):
            if i == 0:
                xiM1 = angles[1]
                xim1 = angle
            elif i == n_samples - 1:
                xiM1 = angle
                xim1 = angles[n_samples - 2]
            else:
                xiM1 = angles[i + 1]
                xim1 = angles[i - 1]

            acc = (np.array(xiM1) - 2 * np.array(angle) + np.array(xim1))/1
            accelerations.append(acc)

        return np.array(accelerations)


    def getInertias(self, positions):

        mass = self._manipulator.getMass()

        inertias = []

        for pos in positions:
            r_1 = pos[0:3]
            r_2 = pos[3:6]
            r_3 = pos[6:9]

            I_1 = mass[0] * (r_1[0] ** 2 + r_1[1] ** 2) + mass[1] * (r_2[0] ** 2 + r_2[1] ** 2) + mass[2] * (r_3[0] ** 2 + r_3[1] ** 2)
            I_2 = mass[0] * (r_1[0] ** 2 + r_1[2] ** 2) + mass[1] * (r_2[0] ** 2 + r_2[2] ** 2) + mass[2] * (r_3[0] ** 2 + r_3[2] ** 2)
            I_3 = mass[1] * np.linalg.norm(r_2 - r_1) ** 2 + mass[2] * np.linalg.norm(r_3 - r_1) ** 2
            I_4 = mass[2] * np.linalg.norm(r_3 - r_2) ** 2

            inertias.append([I_1, I_2, I_3, I_4])

        return inertias

    def getTorques(self, angularAccelerations, inertias):

        return angularAccelerations * inertias


                    