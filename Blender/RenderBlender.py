import os
import json
import time


def render():
    render_model_name = input("Enter the model name: ")
    render_last = False if input("Render all individuals for each model (N: render last individual)? (Y/N): ") == "Y" else True
    render_all = True if input("Render all models (N: render last model or animate=true)? (Y/N): ") == "Y" else False
    render_true = True if not render_all and input("Render only the animate=true individual? (Y/N): ") == "Y" else False


    fps = 30

    with open(f"../Model/Trained Models/{render_model_name}/{render_model_name}.json") as f:
        the_json = json.load(f)

    with open("../Blender/BlenderConfig.json") as f:
        blender_config = json.load(f)

    individuals = the_json["Best Individuals"] if render_all else [the_json["Best Individuals"][-1]]
    if render_true:
        individuals = []
        for ind in the_json["Best Individuals"]:
            if ind["Animate"]:
                individuals.append(ind)
                ind["Animate"] = False
        with open(f"../Model/Trained Models/{render_model_name}/{render_model_name}.json", 'w') as f:
            json.dump(the_json, f)

    for ind in individuals:
        genes = [ind["Genes"][-1]] if render_last else ind["Genes"]
        for gene in genes:
            with open("../Blender/BlenderConfig.json", "w") as f:
                tot_time = ind["Info"]["total_time"]
                blender_config["Desired Position"] = ind["Info"]["desired_position"]
                blender_config["Total time"] = tot_time
                blender_config["Genes to Animate"] = gene[0]
                json.dump(blender_config, f)
            if not os.path.exists(f"../Model/Trained Models/Renders/{ind['ID']}"):
                os.makedirs(f"../Model/Trained Models/Renders/{ind['ID']}")
            render_path = f"{render_model_name}/Renders/{ind['ID']}/individual_{ind['ID']}_gen_{gene[1]}_"
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