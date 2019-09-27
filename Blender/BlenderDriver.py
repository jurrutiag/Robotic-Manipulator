import bpy
from math import radians
import numpy as np


class BlenderDriver:

    def __init__(self, thetas, target, size, end_to_end=False, fps=30, seg=5):
        self._thetas = thetas
        self._target = target
        self._size = size
        self._dimension2scale = 1 / 2
        self._end_to_end = end_to_end

        self._frame_jump = int(np.ceil(fps * seg / len(np.transpose(thetas)[0])))

        self._a1_prev_theta = np.array([0, 0, 0])
        self._a2_prev_theta = np.array([0, 0, 0])
        self._a3_prev_theta = np.array([0, 0, 0])
        self._a4_prev_theta = np.array([0, 0, 0])

    def execute(self):
        A1 = bpy.data.objects["Aone"]
        A2 = bpy.data.objects["Atwo"]
        A3 = bpy.data.objects["Athree"]

        L1 = bpy.data.objects["Lone"]
        L2 = bpy.data.objects["Ltwo"]
        L3 = bpy.data.objects["Lthree"]
        L4 = bpy.data.objects["Lfour"]

        TargetElement = bpy.data.objects["Target"]

        ## Initial configurations

        A1.animation_data_clear()
        A2.animation_data_clear()
        A3.animation_data_clear()

        frame_num = 0

        bpy.context.scene.frame_set(frame_num)

        A1.rotation_euler = self._a1_prev_theta
        A2.rotation_euler = self._a2_prev_theta
        A3.rotation_euler = self._a3_prev_theta

        L1.scale = (0.5, 0.5, self._size[0] * self._dimension2scale)
        L2.scale = (0.5, 0.5, self._size[1] * self._dimension2scale)
        L3.scale = (0.5, 0.5, self._size[2] * self._dimension2scale)
        L4.scale = (0.5, 0.5, self._size[3] * self._dimension2scale)

        # A2.location = (0, 0, -self._size[0])
        # A3.location = (0, 0, -self._size[1])

        TargetElement.location = (self._target[0], self._target[1], self._target[2])

        A1.keyframe_insert(data_path="rotation_euler", index=-1)
        A2.keyframe_insert(data_path="rotation_euler", index=-1)
        A3.keyframe_insert(data_path="rotation_euler", index=-1)

        TargetElement.keyframe_insert(data_path="location", index=-1)


        if self._end_to_end:
            bpy.context.scene.frame_set(20)


            theta = self._thetas

            A1.rotation_euler = [-theta[1], 0, -theta[0]]
            A2.rotation_euler = [-theta[2], 0, 0]
            A3.rotation_euler = [-theta[3], 0, 0]

            A1.keyframe_insert(data_path="rotation_euler", index=-1)
            A2.keyframe_insert(data_path="rotation_euler", index=-1)
            A3.keyframe_insert(data_path="rotation_euler", index=-1)

            frame_num += self._frame_jump

            bpy.context.scene.frame_set(frame_num)
        else:
            # frame_num += 20
            for theta in self._thetas:
                frame_num += self._frame_jump
                bpy.context.scene.frame_set(frame_num)

                A1.rotation_euler = [-theta[1], 0, -theta[0]]
                A2.rotation_euler = [-theta[2], 0, 0]
                A3.rotation_euler = [-theta[3], 0, 0]

                A1.keyframe_insert(data_path="rotation_euler", index=-1)
                A2.keyframe_insert(data_path="rotation_euler", index=-1)
                A3.keyframe_insert(data_path="rotation_euler", index=-1)





if __name__ == "__main__":
    import sys

    sys.path.insert(1, 'D:/Docs universidad/8vo Semestre/Inteligencia Computacional/Robotic Manipulator Project/Model')
    from GeneticAlgorithm import GeneticAlgorithm
    import numpy as np
    import RoboticManipulator

    # np.random.seed(0)  # for testing

    desired_position = [3, 3, 3]
    manipulator_dimensions = [5, 5, 5, 5]
    manipulator_mass = [1, 1, 1]

    manipulator = RoboticManipulator.RoboticManipulator(manipulator_dimensions, manipulator_mass)
    GA = GeneticAlgorithm(desired_position, manipulator)

    GA.initialization()
    individual = GA.getPopulation()[0]
    final_angle = individual.getFinalAngle()

    angles = np.degrees(final_angle)

    target_position = GA.getManipulator().anglesToPositions(angles[0], angles[1], angles[2], angles[3])[2]

    test_individual = np.array([[0, 0, 0, 0],
                                                [1.52806695, 0.54068567, 0.36379948, 0.16970643],
                                                [1.49939901, 0.49333106, -0.1488213, 0.6959088],
                                                [1.37270402, 0.17394161, -0.22605963, 1.1498158],
                                                [0.71754112, -0.35348272, -0.21600229, 0.94629979],
                                                [0.65161271, -0.29210517, -0.16539527, 0.55971057],
                                                [0.9543095, 0.29641347, 0.07270162, 0.30218822],
                                                [0.82934049, -0.10780598, 0.51588552, 0.10013498],
                                                [1.02374677, -0.89286053, 0.78959481, -0.15520973],
                                                [1.57079633, -1.27391787, 0.8598368, -0.46579067],
                                                [1.57079633, -0.79763171, 0.65498013, - 0.00662622],
                                                [1.57079633, 0.13910032, 0.14997636, 0.4544847],
                                                [1.57079633, 0.91816207, -0.26795858, 0.480443],
                                                [1.00852775, 1.33527716, -0.65027739, 0.62448658],
                                                [0.34289373, 1.48872963, -0.75112997, 0.85599982],
                                                [-0.00596442, 1.50145471, -0.15272399, 1.07252486],
                                                [0.0703819, 1.49861617, 0.67564019, 1.25883525],
                                                [0.66510533, 1.18330734, 1.18284731, 1.36201468],
                                                [1.08709204, 0.54257659, 1.39615603, 1.38803122],
                                                [1.18089955, 0.14064053, 1.40766571, 1.33657903]])

    # driver = BlenderDriver(individual.getGenes(), target_position, manipulator_dimensions)
    driver = BlenderDriver(test_individual, [0, 0, 17.07],
                           [5, 5, 5, 5])

    driver.execute()
