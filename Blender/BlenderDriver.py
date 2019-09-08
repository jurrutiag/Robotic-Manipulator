import bpy
from math import radians


class BlenderDriver:

    def __init__(self, theta, target, size):
        self._theta = theta
        self._target = target
        self._size = size
        self._dimension2scale = 1/2

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

        A1.rotation_euler = (0, 0, 0)
        A2.rotation_euler = (0, 0, 0)
        A3.rotation_euler = (0, 0, 0)

        L1.scale = (0.5, 0.5, self._size[0] * self._dimension2scale)
        L2.scale = (0.5, 0.5, self._size[1] * self._dimension2scale)
        L3.scale = (0.5, 0.5, self._size[2] * self._dimension2scale)

        A1.location = (0, 0, -self._size[0])
        A2.location = (0, 0, -self._size[1])
        A3.location = (0, 0, self._size[2])

        TargetElement.location = (self._target[0], self._target[1], self._target[2])

        A1.keyframe_insert(data_path="rotation_euler", index=-1)
        A2.keyframe_insert(data_path="rotation_euler", index=-1)
        A3.keyframe_insert(data_path="rotation_euler", index=-1)

        TargetElement.keyframe_insert(data_path="location", index=-1)

        rotation1 = [(0, 0, radians(theta_1)), A1]
        rotation2 = [(radians(theta_2), 0, radians(theta_1)), A1]
        rotation3 = [(radians(theta_3), 0, 0), A2]
        rotation4 = [(radians(theta_4), 0, 0), A3]

        rotations = (rotation1, rotation2, rotation3, rotation4)



        for rotation in rotations:
            frame_num += 10

            bpy.context.scene.frame_set(frame_num)
            rotation[1].rotation_euler = rotation[0]
            A1.keyframe_insert(data_path="rotation_euler", index=-1)
            A2.keyframe_insert(data_path="rotation_euler", index=-1)
            A3.keyframe_insert(data_path="rotation_euler", index=-1)


if __name__ == "__main__":
    driver = BlenderDriver((90, -30, 10, -20), (20, 10, 10), (10, 5, 5, 5))
    driver.execute()
