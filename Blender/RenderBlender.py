import os
import json
import time


def render(render_model_name, render_run, render_individuals):

    fps = 30

    with open(f"../Model/Trained Models/{render_model_name}/{render_model_name}.json") as f:
        the_json = json.load(f)

    with open("../Blender/BlenderConfig.json") as f:
        blender_config = json.load(f)

    # runs = the_json["Best Individuals"] if run_selected else [the_json["Best Individuals"][-1]]
    runs = the_json["Best Individuals"] if (render_run == -1) else [the_json["Best Individuals"][render_run]]
    # TODO: Delete this
    if False:
        runs = []
        for ind in the_json["Best Individuals"]:
            if ind["Animate"]:
                runs.append(ind)
                ind["Animate"] = False
        with open(f"../Model/Trained Models/{render_model_name}/{render_model_name}.json", 'w') as f:
            json.dump(the_json, f)

    for run in runs:
        # individuals = [run["Genes"][-1]] if runs_to_render else run["Genes"]
        individuals = [run["Genes"][i] for i in render_individuals]
        for ind in individuals:
            with open("../Blender/BlenderConfig.json", "w") as f:
                tot_time = run["Info"]["total_time"]
                blender_config["Desired Position"] = run["Info"]["desired_position"]
                blender_config["Total time"] = tot_time
                blender_config["Genes to Animate"] = ind[0]
                json.dump(blender_config, f)
            if not os.path.exists(f"../Model/Trained Models/Renders/{run['ID']}"):
                os.makedirs(f"../Model/Trained Models/Renders/{run['ID']}")
            render_path = f"{render_model_name}/Renders/{run['ID']}/individual_{run['ID']}_gen_{ind[1]}_"
            end_frame = fps * (tot_time + 1)

            renderWithPath(render_path, end_frame)


def renderWithPath(render_path, end_frame):
    print(f"Rendering {render_path}...")
    t0 = time.time()
    render_base_command = f'blender ../Blender/mano_robotica.blend --background --python ../Blender/BlenderDriver.py -o \"//../Model/Trained Models/{render_path}\" -e {end_frame} -t 4 -a 1>nul'
    os.system(render_base_command)
    print(f"Time taken: {time.time() - t0} s")

if __name__ == "__main__":
    render()