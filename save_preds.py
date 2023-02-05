import os
import numpy as np
import torch
from tqdm.auto import tqdm
from mico_competition import ChallengeDataset, load_cifar10, load_purchase100, load_model
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--challenge', type=str, required=True, choices=['cifar10', 'purchase100'])

args = parser.parse_args()

CHALLENGE = args.challenge
LEN_TRAINING = 50000
LEN_CHALLENGE = 100

scenarios = os.listdir(CHALLENGE)
phases = ['dev', 'final', 'train']

dataset = load_cifar10(dataset_dir=".") if CHALLENGE == 'cifar10' else load_purchase100(dataset_dir='.')
criterion = torch.nn.CrossEntropyLoss(reduction='none')

for scenario in tqdm(scenarios, desc="scenario"):
    for phase in tqdm(phases, desc="phase"):
        root = os.path.join(CHALLENGE, scenario, 'train')
        for model_folder in tqdm(sorted(os.listdir(root), key=lambda d: int(d.split('_')[1])), desc=f"train models"):
            path = os.path.join(root, model_folder)
            model = load_model(CHALLENGE, path)

            predictions = dict()
            phase_path = os.path.join(CHALLENGE, scenario, phase)

            for mf in tqdm(sorted(os.listdir(phase_path), key=lambda d: int(d.split('_')[1])), desc=f"challenge points in {phase}"):
                phase_path_model = os.path.join(phase_path, mf)

                challenge_dataset = ChallengeDataset.from_path(phase_path_model, dataset=dataset, len_training=LEN_TRAINING)
                challenge_points = challenge_dataset.get_challenges()

                challenge_dataloader = torch.utils.data.DataLoader(
                    torch.utils.data.ConcatDataset(challenge_points),
                    batch_size=2*LEN_CHALLENGE
                )

                features, labels = next(iter(challenge_dataloader))
                output = model(features).detach().numpy()

                predictions[mf] = output

            np.save(f'predictions_{phase}_{scenario}/{model_folder}', predictions)
