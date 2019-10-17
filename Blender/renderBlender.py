import os
import json

render_model_name = input("Enter the model name: ")

end_frame = 180

with open(f"../Model/Trained Models/{render_model_name}/{render_model_name}.json") as f:
    the_json = json.load(f)
    for ind in the_json["Best Individuals"]:
        ind["Animate"] = True

with open(f"../Model/Trained Models/{render_model_name}/{render_model_name}.json", 'w') as f:
    json.dump(the_json, f)

for ind in the_json["Best Individuals"]:
    render_path = f"{render_model_name}/Renders/individual_{ind['ID']}"
    render_base_command = f'blender mano_robotica.blend --background --python BlenderDriver.py -o \"//../Model/Trained Models/{render_path}\" -e {end_frame} -t 4 -a'
    os.system(render_base_command)
