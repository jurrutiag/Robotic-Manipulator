import bpy


class BlenderDriver:

    fps = 30

    def __init__(self, thetas, target, size, end_to_end=False, seg=5):
        self._thetas = thetas
        self._target = target
        self._size = size
        self._dimension2scale = 1 / 2
        self._end_to_end = end_to_end

        self._frame_jump = int(np.ceil(BlenderDriver.fps * seg / len(np.transpose(thetas)[0])))

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

    sys.path.insert(1, '../Model')
    import numpy as np
    import RoboticManipulator
    import json
    from definitions import BLENDER_CONFIG_DIR

    # np.random.seed(0)  # for testing
    manipulator_dimensions = [5, 5, 5, 5]
    manipulator_mass = [1, 1, 1]

    with open(BLENDER_CONFIG_DIR) as f:
        config = json.load(f)
        gene = config["Genes to Animate"]
        desired_position = config["Desired Position"]
        total_time = config["Total time"]

    manipulator = RoboticManipulator.RoboticManipulator(manipulator_dimensions, manipulator_mass)
    driver = BlenderDriver(gene, desired_position, manipulator_dimensions, seg=total_time)
    driver.execute()



