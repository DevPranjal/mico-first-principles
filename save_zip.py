import zipfile
import os
from tqdm.auto import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--challenge', type=str, required=True, choices=['cifar10', 'purchase100'])

args = parser.parse_args()

CHALLENGE = args.challenge
scenarios = os.listdir(CHALLENGE)
phases = ['dev', 'final']

with zipfile.ZipFile(f"predictions_{CHALLENGE}.zip", 'w') as zipf:
    for scenario in tqdm(scenarios, desc="scenario"):
        for phase in tqdm(phases, desc="phase"):
            root = os.path.join(CHALLENGE, scenario, phase)
            for model_folder in tqdm(sorted(os.listdir(root), key=lambda d: int(d.split('_')[1])), desc="model"):
                path = os.path.join(root, model_folder)
                file = os.path.join(path, "prediction.csv")
                if os.path.exists(file):
                    zipf.write(file)
                else:
                    raise FileNotFoundError(
                        f"`prediction.csv` not found in {path}. You need to provide predictions for all challenges")
