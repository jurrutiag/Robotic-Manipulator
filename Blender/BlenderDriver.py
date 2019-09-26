import bpy
from math import radians
import numpy as np

class BlenderDriver:

    def __init__(self, thetas, target, size):
        self._thetas = thetas
        self._target = target
        self._size = size
        self._dimension2scale = 1/2

        self._framejump = 20

        self._a1_prev_theta = np.array([0, 0, 0])
        self._a2_prev_theta = np.array([0, 0, 0])
        self._a3_prev_theta = np.array([0, 0, 0])
        self._a4_prev_theta = np.array([0, 0, 0])

    def execute(self, instant=False):
        A1 = bpy.data.objects["Aone"]
        A2 = bpy.data.objects["Atwo"]
        A3 = bpy.data.objects["Athree"]
        A4 = bpy.data.objects["Afour"]

        L1 = bpy.data.objects["Lone"]
        L2 = bpy.data.objects["Ltwo"]
        L3 = bpy.data.objects["Lthree"]
        L4 = bpy.data.objects["Lfour"]

        TargetElement = bpy.data.objects["Target"]

        ## Initial configurations

        A1.animation_data_clear()
        A2.animation_data_clear()
        A3.animation_data_clear()
        A4.animation_data_clear()

        frame_num = 0

        bpy.context.scene.frame_set(frame_num)

        A1.rotation_euler = self._a1_prev_theta
        A2.rotation_euler = self._a2_prev_theta
        A3.rotation_euler = self._a3_prev_theta
        A3.rotation_euler = self._a4_prev_theta

        L1.scale = (0.5, 0.5, self._size[0] * self._dimension2scale)
        L2.scale = (0.5, 0.5, self._size[1] * self._dimension2scale)
        L3.scale = (0.5, 0.5, self._size[2] * self._dimension2scale)
        L4.scale = (0.5, 0.5, self._size[3] * self._dimension2scale)

        A2.location = (0, 0, -self._size[0])
        A3.location = (0, 0, -self._size[1])
        A4.location = (0, 0, self._size[2])

        TargetElement.location = (self._target[0], self._target[1], self._target[2])

        A1.keyframe_insert(data_path="rotation_euler", index=-1)
        A2.keyframe_insert(data_path="rotation_euler", index=-1)
        A3.keyframe_insert(data_path="rotation_euler", index=-1)
        A4.keyframe_insert(data_path="rotation_euler", index=-1)

        TargetElement.keyframe_insert(data_path="location", index=-1)

        # rotation1 = [np.array([0, 0, radians(theta_1)]) + self._a1inicial, A1]
        # rotation2 = [np.array([0, radians(theta_2), 0]) + self._a1inicial, A2]
        # rotation3 = [np.array([0, radians(theta_3), 0]) + self._a2inicial, A3]
        # rotation4 = [np.array([0, radians(theta_4), 0]) + self._a3inicial, A4]

        # rotations = (rotation1, rotation2, rotation3, rotation4)

        # frame_num += self._framejump
        #
        # bpy.context.scene.frame_set(frame_num)
        #
        # for rotation in rotations:
        #
        #     rotation[1].rotation_euler = rotation[0]
        #
        #     A1.keyframe_insert(data_path="rotation_euler", index=-1)
        #     A2.keyframe_insert(data_path="rotation_euler", index=-1)
        #     A3.keyframe_insert(data_path="rotation_euler", index=-1)
        #     A4.keyframe_insert(data_path="rotation_euler", index=-1)
        #
        #     if not instant:
        #         frame_num += self._framejump
        #         bpy.context.scene.frame_set(frame_num)

        for theta in self._thetas:
            frame_num += 1

            A1.rotation_euler = [0, 0, theta[0]]
            A2.rotation_euler = [0, theta[1], 0]
            A3.rotation_euler = [0, theta[2], 0]
            A4.rotation_euler = [0, theta[3], 0]

if __name__ == "__main__":
    import GeneticAlgorithm
    import numpy as np
    import sys

    sys.path.insert(1, 'D:/Docs universidad/8vo Semestre/Inteligencia Computacional/Robotic Manipulator Project/Model')

    import RoboticManipulator

    np.random.seed(0)

    ga = GeneticAlgorithm.GeneticAlgorithm(desired_position=[3, 0, 0])

    ga.initialization()

    print("Individual:")
    individual = ga.getPopulation()[0]
    print(individual.getGenes()[:-1])

    final_angle = individual.getFinalAngle()

    rb = RoboticManipulator.RoboticManipulator(1, 1, 1, 1)

    angles = np.degrees(final_angle)

    rb.anglesToPosition(angles[0], angles[1], angles[2], angles[3])

    driver = BlenderDriver(angles, rb.getPosition(), (5, 5, 5, 5))
    driver.execute()
