import scipy
from scipy import stats
import os
import numpy as np
import torch
import csv
from torchvision import transforms
from tqdm.auto import tqdm
from mico_competition import ChallengeDataset, load_purchase100, load_model, load_cifar10
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--challenge', type=str, required=True,choices=['cifar10', 'purchase100'])
parser.add_argument('--logit_scaling', action='store_true')
parser.add_argument('--stable_logit_scaling', action='store_true')
parser.add_argument('--global_variance', action='store_true')
parser.add_argument('--online_attack', action='store_true')

args = parser.parse_args()

CHALLENGE = args.challenge
LEN_TRAINING = 50000
LEN_CHALLENGE = 100

dataset = load_cifar10(dataset_dir=".") if CHALLENGE == 'cifar10' else load_purchase100(dataset_dir='.')
criterion = torch.nn.CrossEntropyLoss(reduction='none')

scenarios = os.listdir(CHALLENGE)
phases = ['dev', 'final', 'train']

##########################################
# store training indices for train phase #
##########################################

from collections import defaultdict
train_sets = defaultdict(dict)
for scenario in tqdm(scenarios, desc="scenario"):
    root = os.path.join(CHALLENGE, scenario, 'train')
    for model_folder in tqdm(sorted(os.listdir(root), key=lambda d: int(d.split('_')[1])), desc="model"):
        path = os.path.join(root, model_folder)
        challenge_dataset = ChallengeDataset.from_path(path, dataset=dataset, len_training=LEN_TRAINING)
        challenge_points = challenge_dataset.get_challenges()

        train_sets[scenario][model_folder] = challenge_dataset.member.indices + challenge_dataset.training.indices

##############################
# loading stored predictions #
##############################

predictions = defaultdict(lambda: defaultdict(dict))
trans_predictions = defaultdict(lambda: defaultdict(dict))

for scenario in tqdm(scenarios, desc="scenario"):
    for phase in tqdm(phases, desc="phase"):
        for i in range(100):
            path = os.path.join(f'predictions_{phase}_{scenario}', f'model_{i}.npy')
            predictions[phase][scenario][f'model_{i}'] = np.load(path, allow_pickle=True)[()]

#####################################
# generating membership predictions #
#####################################

for scenario in tqdm(scenarios, desc="scenario"):
    for phase in tqdm(phases, desc="phase"):
        root = os.path.join(CHALLENGE, scenario, phase)
        for model_folder in tqdm(sorted(os.listdir(root), key=lambda d: int(d.split('_')[1])), desc="model"):
            path = os.path.join(root, model_folder)
            challenge_dataset = ChallengeDataset.from_path(path, dataset=dataset, len_training=LEN_TRAINING)
            challenge_points = challenge_dataset.get_challenges()
            challenge_dataloader = torch.utils.data.DataLoader(challenge_points, batch_size=2*LEN_CHALLENGE)

            features, labels = next(iter(challenge_dataloader))
            model_scores = load_model(CHALLENGE, path)(features).detach().numpy()

            scores = []
            means = []
            predicted_scores = []

            scores_in = []
            means_in = []
            predicted_scores_in = []

            pr_out = []

            for i, cp in tqdm(enumerate(challenge_points.indices), desc=f"challenge_points for {model_folder}"):
                models_out = [key for key, val in train_sets[scenario].items() if cp not in val]
                preds_out = np.array([predictions[phase][scenario][m][model_folder][i] for m in models_out])

                ############################################################
                # stable logit scaling / logit scaling / no transformation #
                ############################################################

                if args.stable_logit_scaling:
                    preds_out = preds_out - np.max(preds_out, axis=-1, keepdims=True)
                    preds_out = np.array(np.exp(preds_out), dtype=np.float64)
                    preds_out = preds_out / np.sum(preds_out, axis=-1, keepdims=True)

                    y_true = np.array([p[labels[i]] for p in preds_out])
                    y_wrong = np.sum(preds_out, axis=-1) - y_true

                    preds_out = np.log(y_true + 1e-45) - np.log(y_wrong + 1e-45)
                    mean_out = np.mean(preds_out)
                    std_out = np.std(preds_out)

                    model_pred = model_scores[i]
                    model_pred = model_pred - np.max(model_pred, keepdims=True)
                    model_pred = np.array(np.exp(model_pred), dtype=np.float64)
                    model_pred = model_pred / np.sum(model_pred, keepdims=True)

                    model_pred_true = model_pred[labels[i]]
                    model_pred_wrong = np.sum(model_pred, axis=-1) - model_pred_true

                    model_pred_score = np.log(model_pred_true + 1e-45) - np.log(model_pred_wrong + 1e-45)
                    score = model_pred_score
                elif args.logit_scaling:
                    preds_out = scipy.special.softmax(preds_out, axis=-1)
                    preds_out = [p[labels[i]] for p in preds_out]
                    preds_out = list(map(lambda x: np.log(x / (1 - x + 10e-30)), preds_out))
                    mean_out = np.mean(preds_out, axis=-1)
                    std_out = np.std(preds_out)

                    score = model_scores[i]
                    score = scipy.special.softmax(score)[labels[i]]
                    score = np.log(score / (1 - score + 10e-30))
                else:
                    preds_out = scipy.special.softmax(preds_out, axis=-1)
                    preds_out = [p[labels[i]] for p in preds_out]
                    mean_out = np.mean(preds_out, axis=-1)
                    std_out = np.std(preds_out)

                    score = model_scores[i]
                    score = scipy.special.softmax(score)[labels[i]]

                #################
                # online attack #
                #################

                if args.online_attack:
                    models_in = [key for key, val in train_sets[scenario].items() if cp in val]
                    preds_in = np.array([predictions[phase][scenario][m][model_folder][i] for m in models_in])

                    if args.stable_logit_scaling:
                        raise NotImplementedError('online attack with logit scaling not implemented yet')
                    elif args.logit_scaling:
                        raise NotImplementedError('online attack with logit scaling not implemented yet')
                    else:
                        if len(models_in) == 0:
                            mean_in = mean_out
                            std_in = std_out
                        else:
                            preds_in = scipy.special.softmax(preds_in, axis=-1)
                            preds_in = [p[labels[i]] for p in preds_in]
                            mean_in = np.mean(preds_in, axis=-1)
                            std_in = np.std(preds_in)

                            score_in = model_scores[i]
                            score_in = scipy.special.softmax(score_in)[labels[i]]


                ###################
                # global variance #
                ###################

                if args.global_variance:
                    scores.append(score)
                    means.append(mean_out)
                    predicted_scores.extend(preds_out)

                    if args.online_attack:
                        scores_in.append(score_in)
                        means_in.append(mean_in)
                        predicted_scores_in.extend(preds_in)
                else:
                    if args.online_attack:
                        test_score = scipy.stats.norm.pdf(score, mean_in, std_in+1e-30) / scipy.stats.norm.pdf(score, mean_out, std_out+1e-30)
                    else:
                        test_score = scipy.stats.norm.cdf(score, mean_out, std_out+1e-30)

                    pr_out.append(test_score)


            if args.global_variance:
                if args.online_attack:
                    preds = scipy.stats.norm.pdf(scores_in, means_in, np.std(predicted_scores_in)+1e-30) / scipy.stats.norm.pdf(scores, means, np.std(predicted_scores)+1e-30)
                else:
                    preds = scipy.stats.norm.cdf(scores, means, np.std(predicted_scores)+1e-30)
            else:
                preds = np.array(pr_out)

            if not args.online_attack:
                assert np.all((0 <= preds) & (preds <= 1))
    
            with open(os.path.join(path, "prediction.csv"), "w") as f:
                csv.writer(f).writerow(preds)
