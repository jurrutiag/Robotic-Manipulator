

if __name__ == "__main__":
    import json
    import numpy as np

    with open("Trained Models/128_runs_4_cores_torque_and_no_torque/128_runs_4_cores_torque_and_no_torque.json") as f:
        data = json.load(f)

    times = []
    max = 0
    for model in data["Best Individuals"][64:]:
        times.append(model["Fitness"])
        if model["Fitness"] >= max:
            max = model["Fitness"]
            max_id = model["ID"]

    print(f"Mean: {np.mean(times)}, std: {np.std(times)}, max: {np.max(times)}, id: {max_id}")