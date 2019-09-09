import bpy
from math import radians
import numpy as np

class BlenderDriver:

    def __init__(self, theta, target, size):
        self._theta = theta
        self._target = target
        self._size = size
        self._dimension2scale = 1/2

        self._a1inicial = np.array([0, 0, radians(-90)])
        self._a2inicial = np.array([radians(90), 0, 0])
        self._a3inicial = np.array([radians(-90), 0, 0])

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

        theta_1 = self._theta[0]
        theta_2 = self._theta[1]
        theta_3 = self._theta[2]
        theta_4 = self._theta[3]

        frame_num = 0

        bpy.context.scene.frame_set(frame_num)

        A1.rotation_euler = self._a1inicial
        A2.rotation_euler = self._a2inicial
        A3.rotation_euler = self._a3inicial

        L1.scale = (0.5, 0.5, self._size[0] * self._dimension2scale)
        L2.scale = (0.5, 0.5, self._size[1] * self._dimension2scale)
        L3.scale = (0.5, 0.5, self._size[2] * self._dimension2scale)
        L4.scale = (0.5, 0.5, self._size[3] * self._dimension2scale)

        A1.location = (0, 0, -self._size[0])
        A2.location = (0, 0, -self._size[1])
        A3.location = (0, 0, self._size[2])

        TargetElement.location = (self._target[0], self._target[1], self._target[2])

        A1.keyframe_insert(data_path="rotation_euler", index=-1)
        A2.keyframe_insert(data_path="rotation_euler", index=-1)
        A3.keyframe_insert(data_path="rotation_euler", index=-1)

        TargetElement.keyframe_insert(data_path="location", index=-1)

        rotation1 = [np.array([0, 0, -radians(theta_1)]) + self._a1inicial, A1]
        rotation2 = [np.array([-radians(theta_2), 0, -radians(theta_1)]) + self._a1inicial, A1]
        rotation3 = [np.array([-radians(theta_3), 0, 0]) + self._a2inicial, A2]
        rotation4 = [np.array([-radians(theta_4), 0, 0]) + self._a3inicial, A3]

        rotations = (rotation1, rotation2, rotation3, rotation4)



        for rotation in rotations:
            frame_num += 10

            bpy.context.scene.frame_set(frame_num)
            rotation[1].rotation_euler = rotation[0]
            A1.keyframe_insert(data_path="rotation_euler", index=-1)
            A2.keyframe_insert(data_path="rotation_euler", index=-1)
            A3.keyframe_insert(data_path="rotation_euler", index=-1)


if __name__ == "__main__":
    import sys
    sys.path.insert(1, 'D:/Docs universidad/8vo Semestre/Inteligencia Computacional/Robotic Manipulator Project/Model')

    import RoboticManipulator

    rb = RoboticManipulator.RoboticManipulator(5, 5, 5, 5)

    angles = (45, -10, 10, 15)

    rb.anglesToPosition(angles[0], angles[1], angles[2], angles[3])

    driver = BlenderDriver(angles, rb.getPosition(), (5, 5, 5, 5))
    driver.execute()
