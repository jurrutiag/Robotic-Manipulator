import os
import json
import time
from definitions import getModelDir, BLENDER_CONFIG_DIR, MODEL_TRAININGS_DIR, BLEND_FILE_DIR, BLENDER_DRIVER_DIR

def render(render_model_name, render_run, render_individuals, all_inds=False):

    fps = 30

    with open(getModelDir(render_model_name)) as f:
        the_json = json.load(f)

    with open(BLENDER_CONFIG_DIR) as f:
        blender_config = json.load(f)

    # runs = the_json["Best Individuals"] if run_selected else [the_json["Best Individuals"][-1]]
    runs = the_json["Best Individuals"] if (render_run == -1) else [the_json["Best Individuals"][render_run]]

    for run in runs:

        # individuals = [run["Genes"][-1]] if runs_to_render else run["Genes"]
        individuals = [run["Genes"][i] for i in render_individuals] if (render_run != -1) else run["Genes"]
        for ind in individuals:
            with open(BLENDER_CONFIG_DIR, "w") as f:
                tot_time = run["Info"]["total_time"]
                blender_config["Desired Position"] = run["Info"]["desired_position"]
                blender_config["Total time"] = tot_time
                blender_config["Genes to Animate"] = ind[0]
                json.dump(blender_config, f)
            if not os.path.exists(MODEL_TRAININGS_DIR + f"/Renders/{run['ID']}"):
                os.makedirs(os.path.join(MODEL_TRAININGS_DIR, 'Renders', str(run['ID'])))
            render_path = f"{render_model_name}/Renders/{run['ID']}/individual_{run['ID']}_gen_{ind[1]}_"
            end_frame = fps * (tot_time + 1)

            renderWithPath(render_path, end_frame)


def renderWithPath(render_path, end_frame):
    print(f"Rendering {render_path}...")
    t0 = time.time()
    render_base_command = f'blender \"{BLEND_FILE_DIR}\" --background --python \"{BLENDER_DRIVER_DIR}\" -o \"{os.path.join(MODEL_TRAININGS_DIR, render_path)}\" -e {end_frame} -t 4 -a 1>nul'
    os.system(render_base_command)
    print(f"Time taken: {time.time() - t0} s")

if __name__ == "__main__":
    render()