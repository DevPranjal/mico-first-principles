# Microsoft Membership Inference Competition: 2nd Place Solution

### Solution

We follow the methodology used by Carlini et. al. in their work [Membership Inference Attacks From First Principles](https://arxiv.org/abs/2112.03570). To achieve this, we reproduce the code from scratch to fit in the data provided by the competition authors. Carlini et. al. summarize a few tricks which helped them achieve state of the art TPR at low FPRs. We implemented these 'tricks' but were not able to benefit from most of them. We believe that this is because of a very low number of shadow models available (more details below).

The 'tricks' mentioned by Carlini et. al. are listed below. Alongside, we also state the obervations on applying / removing these.

- [ ] _Logit Scaling / Stable Logit Scaling_: This scales the logit scores to help them resemble a gaussian distribution. However, since the number of `models_out` (models in 'train' phase not trained on a certain challenge point, refer paper and code for more details) were around 20 compared to the results in the paper which have 256, the gaussian could not be well formed. Note that we did not train any additional models other than those provided in the data files.
- [ ] _Online Attack_: This uses both `models_out` and `models_in` to predict inference scores. On turning this feature on, our TPR was impacted negatively. We again reason this to poorly formed gaussians due to low and uneven number of `models_out` and `models_in`.
- [x] _Global Variance_: This uses global variance of all challenge points for a model to fit the gaussians. This helped us improve our results as expected (global variance helps with small number of shadow models according to the paper).
- [ ] _Augmentations_: Uses inference on augmented versions of challenge points to fit a multivariate gaussian. Since models trained by the challenge authors did not include random augmentations, this would not help. Our claim is supported by experiments.

Further, we include some of the methods which we tried but are not part of the final solution. These methods include one using the entropy of the outputs and another using a neural network classifier which analyses the output and loss distributions to determine membership of a sample. These methods are possible directions for further research in the area.

### Steps to reproduce submitted results

**For CIFAR-10 Track**

| Command | Purpose |
| --- | --- |
| `python download_files.py --challenge 'cifar10'` | Downloads files provided by the challenge authors |
| `python save_preds.py --challenge 'cifar10'` | Stores predictions of 'train' models on challenge points offline for faster membership inference predictions |
| `python run_lira_mia.py --challenge 'cifar10' --global_variance`| Stores membership inference scores for all challenge points in respective 'prediction.csv' |
| `python save_zip.py --challenge 'cifar10'` | Saves inference scores as zip file for submission |

**For Purchase-100 Track**

| Command | Purpose |
| --- | --- |
| `python download_files.py --challenge 'purchase100'` | Downloads files provided by the challenge authors |
| `python save_preds.py --challenge 'purchase100'` | Stores predictions of 'train' models on challenge points offline for faster membership inference predictions |
| `python run_lira_mia.py --challenge 'purchase100' --global_variance`| Stores membership inference scores for all challenge points in respective 'prediction.csv' |
| `python save_zip.py --challenge 'purchase100'` | Saves inference scores as zip file for submission |

_Note_:

1. `run_lira_mia.py` has other options: `--global_variance`, `--logit_scaling`, `--stable_logit_scaling`, `--online_attack`
2. `--logit_scaling` and `--stable_logit_scaling` are not implemented with `--online_attack`
