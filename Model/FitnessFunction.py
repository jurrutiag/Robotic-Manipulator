import RoboticManipulator
import numpy as np
import math

class FitnessFunction:

    def __init__(self, manipulator, torques_ponderations, desired_position, sampling_points, total_time=5, distance_error_ponderation=1, torques_error_ponderation=1):

        self._manipulator = manipulator
        self._torques_ponderations = torques_ponderations
        self._desired_position = desired_position
        self._distance_error_ponderation = distance_error_ponderation
        self._torques_error_ponderation = torques_error_ponderation
        self._delta_t = 1

        self._torque = 0
        self._dist = 0

    def evaluateFitness(self, individual):

        positions = self.getPositions(individual)
        distance_error = np.linalg.norm(self._desired_position - positions[-1][3], ord=2)

        if self._torques_error_ponderation != 0:
            angAccelerations = self.getAngularAccelerations(individual)

            inertias = self.getInertias(positions)
            torques = self.getTorques(angAccelerations, inertias, positions)

            sum_torques = np.linalg.norm(torques, ord=1, axis=0)

            torques_error = sum_torques @ self._torques_ponderations
        else:
            torques_error = 0

        fitness = 1 / (1 + self._torques_error_ponderation * torques_error + self._distance_error_ponderation * distance_error)

        individual.setFitness(fitness)
        self._torque = torques_error
        self._dist = distance_error

    #retorna posiciones cartesionas de las tres masas móviles del brazo
    def getPositions(self, individual):

        # a2x, a2y, a2z, a3x, a3y, a3z, a4x, a4y, a4z

        positions = []

        for ang in individual.getGenes():

            theta_1, theta_2, theta_3, theta_4 = ang

            thisPosition = self._manipulator.anglesToPositions(theta_1, theta_2, theta_3, theta_4)

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

            acc = (np.array(xiM1) - 2 * np.array(angle) + np.array(xim1))/(self._delta_t ** 2)
            accelerations.append(acc)

        return np.array(accelerations)


    def getInertias(self, positions):

        mass = self._manipulator.getMass()

        inertias = []

        for pos in positions:
            r_1 = pos[0]
            r_2 = pos[1]
            r_3 = pos[2]
            r_4 = pos[3]

            I_1 = mass[0] * (r_2[0] ** 2 + r_2[1] ** 2) + mass[1] * (r_3[0] ** 2 + r_3[1] ** 2) + mass[2] * (r_4[0] ** 2 + r_4[1] ** 2)
            I_2 = mass[0] * np.linalg.norm(r_2 - r_1) ** 2 + mass[1] * np.linalg.norm(r_3 - r_1) ** 2 + mass[2] * np.linalg.norm(r_4 - r_1) ** 2
            I_3 = mass[1] * np.linalg.norm(r_3 - r_2) ** 2 + mass[2] * np.linalg.norm(r_4 - r_2) ** 2
            I_4 = mass[2] * np.linalg.norm(r_4 - r_3) ** 2

            inertias.append([I_1, I_2, I_3, I_4])

        return inertias

    def getTorques(self, angularAccelerations, inertias, positions):

        g = 9.8
        gravity_torques = []
        mass = self._manipulator.getMass()

        for pos in positions:
            r_2 = pos[1]
            r_3 = pos[2]
            r_4 = pos[3]

            mass_center1 = (r_2 * mass[0] + r_3 * mass[1] + r_4 * mass[2]) / sum(mass)
            mass_center2 = ((r_3 - r_2) * mass[1] + (r_4 - r_2) * mass[2]) / (mass[1] + mass[2])
            mass_center3 = r_4 - r_3

            force_1 = np.array([0, 0, -sum(mass) * g])
            force_2 = np.array([0, 0, -(mass[1] + mass[2]) * g])
            force_3 = np.array([0, 0, -mass[2] * g])

            gravity_torque_1 = np.linalg.norm(np.cross(mass_center1, force_1), ord=2)
            gravity_torque_2 = np.linalg.norm(np.cross(mass_center2, force_2), ord=2)
            gravity_torque_3 = np.linalg.norm(np.cross(mass_center3, force_3), ord=2)

            gravity_torques.append([0, gravity_torque_1, gravity_torque_2, gravity_torque_3])

        return angularAccelerations * inertias - gravity_torques

    def getTorques(self):
        return self._torque

    def getDistance(self):
        return self._dist

