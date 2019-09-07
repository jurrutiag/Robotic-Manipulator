import bpy
from math import radians


class BlenderDriver:

    def __init__(self, theta):
        self._theta = theta

    def execute(self):
        A1 = bpy.data.objects["Aone"]
        A2 = bpy.data.objects["Atwo"]
        A3 = bpy.data.objects["Athree"]

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

        A1.keyframe_insert(data_path="rotation_euler", index=-1)
        A2.keyframe_insert(data_path="rotation_euler", index=-1)
        A3.keyframe_insert(data_path="rotation_euler", index=-1)

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
    driver = BlenderDriver((90, -30, 10, -20))
    driver.execute()
