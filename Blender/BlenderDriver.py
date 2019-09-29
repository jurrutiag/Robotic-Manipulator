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

            A1.rotation_euler = [0, theta[1], -theta[0]]
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

                A1.rotation_euler = [0, theta[1], -theta[0]]
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
    import json

    # np.random.seed(0)  # for testing

    desired_position = [3, 3, 3]
    manipulator_dimensions = [5, 5, 5, 5]
    manipulator_mass = [1, 1, 1]

    with open("D:/Docs universidad/8vo Semestre/Inteligencia Computacional/Robotic Manipulator Project/Model/JSON files/best_individuals.json") as f:
        best_individuals = json.load(f)
        for ind in best_individuals["Best Individuals"]:
            if ind["Animate"]:
                test_individual = ind["Genes"]
                print(test_individual)

                manipulator = RoboticManipulator.RoboticManipulator(manipulator_dimensions, manipulator_mass)

                driver = BlenderDriver(test_individual, desired_position, manipulator_dimensions)

                driver.execute()
                break
        else:
            print("No individual to animate...")


