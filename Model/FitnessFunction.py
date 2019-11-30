import RoboticManipulator
import numpy as np
import math


class FitnessFunction:

    def __init__(self, manipulator, desired_position, sampling_points, total_time=5, torques_ponderations=(1, 1, 1, 1),
                 distance_error_ponderation=1, torques_error_ponderation=1, velocity_error_ponderation=1):

        self._manipulator = manipulator
        self._torques_ponderations = torques_ponderations
        self._desired_position = desired_position

        # 85.6, 0.13 are the multipliers for distance to get equivalent torque and velocity respectively.
        self._distance_error_ponderation = distance_error_ponderation
        self._torques_error_ponderation = torques_error_ponderation * (1 / 256.79)
        self._velocity_error_ponderation = velocity_error_ponderation * (1 / 0.39)
        self._delta_t = 1

        self._torque = 0
        self._dist = 0

    def evaluateSeparateFitnesses(self, individual):

        positions = self.getPositions(individual)
        distance_error = np.linalg.norm(self._desired_position - positions[-1][3], ord=2)

        angVelocities = self.getAngularVelocities(individual)
        velocity_error = np.linalg.norm(angVelocities)

        angAccelerations = self.getAngularAccelerations(individual)

        inertias = self.getInertias(positions)
        torques = self.getTorques(angAccelerations, inertias, positions, individual)

        torques_error = torques @ np.array(self._torques_ponderations)

        individual.setMultiFitness(np.array([distance_error, torques_error, velocity_error]))
        individual.setFitness(1 / (
                    1 + self._torques_error_ponderation * torques_error + self._distance_error_ponderation * distance_error + self._velocity_error_ponderation * velocity_error))

    #retorna posiciones cartesionas de las tres masas móviles del brazo
    def getPositions(self, individual):

        # a2x, a2y, a2z, a3x, a3y, a3z, a4x, a4y, a4z

        positions = []

        for ang in individual.getGenes():

            theta_1, theta_2, theta_3, theta_4 = ang

            thisPosition = self._manipulator.anglesToPositions(theta_1, theta_2, theta_3, theta_4)

            positions.append(thisPosition)

        return np.array(positions)

    def getAngularVelocities(self, individual):
        angles = individual.getGenes()

        n_samples = len(angles)

        velocities = []
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

            vel = (np.array(xiM1) - np.array(xim1)) / (2 * self._delta_t ** 2)
            velocities.append(vel)

        return np.array(velocities)


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

    def getTorques(self, angularAccelerations, inertias, positions, individual):

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

            # gravity_torque_1 = np.cross(mass_center1, force_1)
            # gravity_torque_2 = np.cross(mass_center2, force_2)
            # gravity_torque_3 = np.cross(mass_center3, force_3)

            # Cross product optimization:

            gravity_torque_1 = [mass_center1[1] * force_1[2], - mass_center1[0] * force_1[2], 0]
            gravity_torque_2 = [mass_center2[1] * force_2[2], - mass_center2[0] * force_2[2], 0]
            gravity_torque_3 = [mass_center3[1] * force_3[2], - mass_center3[0] * force_3[2], 0]

            gravity_torques.append([0, gravity_torque_1, gravity_torque_2, gravity_torque_3])

        # Torque axis

        gravity_torques_array = np.array(gravity_torques)

        angles = individual.getGenes()
        angles_1 = angles[:, 0]
        phi_x = np.expand_dims(np.cos(angles_1 - np.pi / 2), axis=1)
        phi_y = np.expand_dims(np.sin(angles_1 - np.pi / 2), axis=1)

        torque_value = angularAccelerations * inertias

        torque_value_1 = np.expand_dims(torque_value[:, 1], axis=1)
        torque_value_2 = np.expand_dims(torque_value[:, 2], axis=1)
        torque_value_3 = np.expand_dims(torque_value[:, 3], axis=1)

        t_1 = np.array(list(map(lambda x: x * np.array([0, 0, 1]), torque_value[:, 0])))
        t_2 = np.concatenate([phi_x * torque_value_1, phi_y * torque_value_1, np.zeros([len(phi_x), 1])], axis=1)
        t_3 = np.concatenate([phi_x * torque_value_2, phi_y * torque_value_2, np.zeros([len(phi_x), 1])], axis=1)
        t_4 = np.concatenate([phi_x * torque_value_3, phi_y * torque_value_3, np.zeros([len(phi_x), 1])], axis=1)

        t_1_norm = np.linalg.norm(t_1, axis=1, ord=2)
        t_2_norm = np.linalg.norm(t_2, axis=1, ord=2)
        t_3_norm = np.linalg.norm(t_3, axis=1, ord=2)
        t_4_norm = np.linalg.norm(t_4, axis=1, ord=2)

        return np.array([sum(t_1_norm), sum(t_2_norm), sum(t_3_norm), sum(t_4_norm)])

