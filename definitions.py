import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

BLENDER_DIR = os.path.join(ROOT_DIR, 'Blender')
INFODISPLAY_DIR = os.path.join(ROOT_DIR, 'InfoDisplay')
MEDIA_DIR = os.path.join(ROOT_DIR, 'Media')
MODEL_DIR = os.path.join(ROOT_DIR, 'Model')

# Blender:
BLENDER_CONFIG_DIR = os.path.join(BLENDER_DIR, 'BlenderConfig.json')

# Info Display:
PARAMETERS_VARIATIONS_INFO_DIR = os.path.join(INFODISPLAY_DIR, 'parameters_variations.json')

# Model:
MODEL_TRAININGS_DIR = os.path.join(MODEL_DIR, 'Trained Models')
def getModelDir(model_name):
    return os.path.join(MODEL_TRAININGS_DIR, model_name, model_name + '.json')