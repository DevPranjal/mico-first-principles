from scipy import special
from scipy import stats
import os
import numpy as np
import torch
import csv
import torch.nn as nn
import zipfile
from tqdm.auto import tqdm
from mico_competition import ChallengeDataset, load_cifar10, load_model
import math

class meta_network(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(13)
        self.l1 = nn.Linear(13, 10)
        self.relu = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(10)
        self.l2 = nn.Linear(10, 5)
        self.bn3 = nn.BatchNorm1d(5)
        self.l3 = nn.Linear(5, 3)        
        self.bn4 = nn.BatchNorm1d(3)
        self.l4 = nn.Linear(3, 1)
        # self.bn5 = nn.BatchNorm1d(7)
        # self.l5 = nn.Linear(7, 3)
        # self.bn6 = nn.BatchNorm1d(3)
        # self.l6 = nn.Linear(3, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input):
        input = self.bn1(input)
        x = self.l1(input)
        x = self.relu(x)
        x = self.l2(self.bn2(x))
        x = self.relu(x)
        x = self.l3(self.bn3(x))
        x = self.relu(x)
        x = self.l4(self.bn4(x))
        # x = self.relu(x)
        # x = self.l5(self.bn5(x))
        # x = self.relu(x)
        # x = self.l6(self.bn6(x))
        return self.sigmoid(x)

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

meta_net = meta_network()
meta_net.apply(init_weights)
meta_net.train()

CHALLENGE = "cifar10"
LEN_TRAINING = 50000
LEN_CHALLENGE = 100

dataset = load_cifar10(dataset_dir="..")
criterion = torch.nn.CrossEntropyLoss(reduction='none')

###### EDITABLE #######

scenarios = ['cifar10_inf', 'cifar10_hi', 'cifar10_lo']
phases = ['train']

#######################

# store training indices for train phase

from collections import defaultdict
train_sets = defaultdict(dict)
for scenario in tqdm(scenarios, desc="scenario"):
    root = os.path.join(CHALLENGE, scenario, 'train')
    for model_folder in tqdm(sorted(os.listdir(root), key=lambda d: int(d.split('_')[1])), desc="model"):
        path = os.path.join(root, model_folder)
        challenge_dataset = ChallengeDataset.from_path(path, dataset=dataset, len_training=LEN_TRAINING)
        challenge_points = challenge_dataset.get_challenges()

        train_sets[scenario][model_folder] = challenge_dataset.member.indices + challenge_dataset.training.indices

# loading stored predictions
# scenarios = ['cifar10_inf']

predictions = defaultdict(lambda: defaultdict(dict))
for scenario in tqdm(scenarios, desc="scenario"):
    for phase in tqdm(phases, desc="phase"):
        for i in range(100):
            path = os.path.join(f'predictions_{phase}_{scenario[8:]}', f'model_{i}.npy')
            predictions[phase][scenario][f'model_{i}'] = np.load(path, allow_pickle=True)[()]

loss_fn = nn.BCELoss(reduction = 'mean')
optimizer = torch.optim.Adam(meta_net.parameters(), lr = 0.0001)
pr_out = []
num_epochs = 20
for epoch in range(num_epochs):
    for scenario in tqdm(scenarios, desc="scenario"):
        for phase in tqdm(phases, desc="phase"):
            root = os.path.join(CHALLENGE, scenario, phase)
            for model_folder in tqdm(sorted(os.listdir(root), key=lambda d: int(d.split('_')[1])), desc="model"):
                path = os.path.join(root, model_folder)
                meta_labels = torch.tensor(np.loadtxt(os.path.join(path, "solution.csv"),   delimiter=","))
                challenge_dataset = ChallengeDataset.from_path(path, dataset=dataset, len_training=LEN_TRAINING)
                challenge_points = challenge_dataset.get_challenges()
                challenge_dataloader = torch.utils.data.DataLoader(challenge_points, batch_size=2*LEN_CHALLENGE)
    
                features, labels = next(iter(challenge_dataloader))
                model_scores = load_model('cifar10', path)(features).detach().numpy()
                model_scores = special.softmax(model_scores, axis=-1)
    
                pr_in = []
    
                for i, cp in tqdm(enumerate(challenge_points.indices), desc=f"challenge_points for {model_folder}"):
                    models_out = [key for key, val in train_sets[scenario].items() if cp not in val]
                    inp = dataset.__getitem__(cp)[0].unsqueeze(0)
                    preds_out = np.array([predictions[phase][scenario][m][model_folder][i] for m in models_out])
                    preds_out = special.softmax(preds_out, axis=-1)
                    preds_out = [p[labels[i]] for p in preds_out]
                    preds_out = list(map(lambda x: np.log(x / (1 - x + 10e-30)), preds_out))
                    
                    mean_out = np.mean(preds_out)
                    std_out = np.std(preds_out)
    
                    score = model_scores[i][labels[i]]
                    score = np.log(score / (1 - score))
    
                    stats.norm.cdf(score, mean_out, std_out+1e-30)
                    input = torch.Tensor(np.concatenate((np.mean(model_scores, axis=0), np.expand_dims(mean_out, axis = 0), np.expand_dims(std_out, axis = 0), np.expand_dims(score, axis = 0)))).float()
                    pr_in.append(input)
                inputs = torch.stack(pr_in)
                preds = torch.squeeze(meta_net(inputs))
                pr_out.append(preds)
                preds = torch.nan_to_num(preds, nan = 0.5)
                loss = loss_fn(preds, meta_labels.float())
                print(f"[  epoch: {epoch} | loss: {loss.item()}  ]")
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

########################################################################################################################################################################################################################################################################################################
torch.save(meta_net, 'latest_meta_net.pt')
meta_net.eval()
# exit()
scenarios = ['cifar10_inf', 'cifar10_hi', 'cifar10_lo']
phases = ['dev', 'final']

#######################

# store training indices for train phase

from collections import defaultdict
train_sets = defaultdict(dict)
for scenario in tqdm(scenarios, desc="scenario"):
    root = os.path.join(CHALLENGE, scenario, 'train')
    for model_folder in tqdm(sorted(os.listdir(root), key=lambda d: int(d.split('_')[1])), desc="model"):
        path = os.path.join(root, model_folder)
        challenge_dataset = ChallengeDataset.from_path(path, dataset=dataset, len_training=LEN_TRAINING)
        challenge_points = challenge_dataset.get_challenges()

        train_sets[scenario][model_folder] = challenge_dataset.member.indices + challenge_dataset.training.indices

# loading stored predictions
# scenarios = ['cifar10_inf']

predictions = defaultdict(lambda: defaultdict(dict))
for scenario in tqdm(scenarios, desc="scenario"):
    for phase in tqdm(phases, desc="phase"):
        for i in range(100):
            path = os.path.join(f'predictions_{phase}_{scenario[8:]}', f'model_{i}.npy')
            predictions[phase][scenario][f'model_{i}'] = np.load(path, allow_pickle=True)[()]


# generating membership predictions
phases = ['dev', 'final']
preds_all = []
for scenario in tqdm(scenarios, desc="scenario"):
    for phase in tqdm(phases, desc="phase"):
        root = os.path.join(CHALLENGE, scenario, phase)
        for model_folder in tqdm(sorted(os.listdir(root), key=lambda d: int(d.split('_')[1])), desc="model"):
            path = os.path.join(root, model_folder)

            challenge_dataset = ChallengeDataset.from_path(path, dataset=dataset, len_training=LEN_TRAINING)
            challenge_points = challenge_dataset.get_challenges()
            challenge_dataloader = torch.utils.data.DataLoader(challenge_points, batch_size=2*LEN_CHALLENGE)

            features, labels = next(iter(challenge_dataloader))
            model_scores = load_model('cifar10', path)(features).detach().numpy()
            model_scores = special.softmax(model_scores, axis=-1)

            pr_out = []

            for i, cp in tqdm(enumerate(challenge_points.indices), desc=f"challenge_points for {model_folder}"):
                models_out = [key for key, val in train_sets[scenario].items() if cp not in val]
                inp = dataset.__getitem__(cp)[0].unsqueeze(0)

                preds_out = np.array([predictions[phase][scenario][m][model_folder][i] for m in models_out])
                preds_out = special.softmax(preds_out, axis=-1)
                preds_out = [p[labels[i]] for p in preds_out]
                preds_out = list(map(lambda x: np.log(x / (1 - x + 10e-30)), preds_out))
                
                mean_out = np.mean(preds_out)
                std_out = np.std(preds_out)

                score = model_scores[i][labels[i]]
                score = np.log(score / (1 - score))

                stats.norm.cdf(score, mean_out, std_out+1e-30)
                pr_out.append(torch.squeeze(meta_net(torch.unsqueeze(torch.Tensor(np.concatenate((np.mean(model_scores, axis=0), np.expand_dims(mean_out, axis = 0), np.expand_dims(std_out, axis = 0), np.expand_dims(score, axis = 0)))).float(), dim=0))).detach().numpy())
                
            preds = np.array(pr_out)
            preds = np.nan_to_num(preds, nan = 0.5)
            preds_out.append(preds)
            for k in range(len(preds)):
                if preds[k] < 0.6:
                    preds[k] = 0
            assert np.all((0 <= preds) & (preds <= 1))
            with open(os.path.join(path, "prediction.csv"), "w") as f:
                csv.writer(f).writerow(preds)

print(preds_out)

phases = ['dev', 'final']

with zipfile.ZipFile("predictions_meta.zip", 'w') as zipf:
    for scenario in tqdm(scenarios, desc="scenario"): 
        for phase in tqdm(phases, desc="phase"):
            root = os.path.join(CHALLENGE, scenario, phase)
            for model_folder in tqdm(sorted(os.listdir(root), key=lambda d: int(d.split('_')[1])), desc="model"):
                path = os.path.join(root, model_folder)
                file = os.path.join(path, "prediction.csv")
                if os.path.exists(file):
                    zipf.write(file)
                else:
                    raise FileNotFoundError(f"`prediction.csv` not found in {path}. You need to provide predictions for all challenges")
