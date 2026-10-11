# Deep multi-criteria collaborative filtering

This repository implements Nassar, Jafar, and Rahhal, "A novel deep multi-criteria collaborative filtering model for recommendation system" (Knowledge-Based Systems, 2020).

The model has two networks. The criteria network predicts seven hotel ratings from a user id and a hotel id. The overall network predicts one overall rating from those seven ratings. The two networks train separately. Prediction uses the criteria network first, then the overall network.

The hotel reviews are in `texts.rar`. That archive is the TripAdvisor file from [Nour-Nassar/Datasets](https://github.com/Nour-Nassar/Datasets). A row stays in the data only when all eight ratings are integers from 1 to 5. The filter gives 81,085 ratings, 72,119 users, and 1,850 hotels.

## Setup

Create the `rds` environment from `environment.yml`. Do not install these packages into the base environment.

```bash
conda env create -f environment.yml
conda run -n rds pip install torch --index-url https://download.pytorch.org/whl/cu130
```

PyTorch is a second step. `environment.yml` does not pass the CUDA package index. `ipykernel` is in the environment file, so the kernel command below works after the environment exists.

If the CUDA wheel does not install, install the CPU wheel:

```bash
conda run -n rds pip install torch
```

Register the notebook kernel:

```bash
conda run -n rds python -m ipykernel install --user --name rds --display-name "Python (rds)"
```

## Run the experiment

From the repository root:

```bash
conda run --no-capture-output -n rds python -u -m src.experiment
```

The run uses five folds. Each fold keeps 20 percent of the ratings as the test set. One percent of all ratings is the validation set. The rest is the fit set. Neural networks train on the fit set and stop early on the validation set. The classical baselines train on the fit set and the validation set together.

The command accepts these flags:

| Flag | Default | Role |
| --- | --- | --- |
| `--archive` | `texts.rar` | Hotel review archive |
| `--folds` | `5` | Number of folds |
| `--max-epochs` | `30` | Epoch limit |
| `--patience` | `5` | Epochs with no validation gain before a stop |
| `--top-k` | `10` | Cutoff for F1, F2, MAP, and MRR |
| `--batch-size` | `512` | Mini-batch size |
| `--seed` | `42` | Shuffle seed and network seed |
| `--device` | CUDA if available, else CPU | Torch device |
| `--results` | `results/fold_metrics.json` | Metric file |
| `--log` | `results/experiment.log` | Training log |

The log appends. It does not replace earlier text.

## Notebooks

Open the notebooks with the `rds` kernel.

| Notebook | Content |
| --- | --- |
| `notebooks/01_data.ipynb` | Table 1 counts, score counts, and the rating chart |
| `notebooks/02_model.ipynb` | The three networks and the 48-row overfit check |
| `notebooks/03_results.ipynb` | Five-fold MAE next to the paper, plus the charts |
| `notebooks/04_experiment.ipynb` | One fold. Change the settings cell, then run the notebook again |

`04_experiment.ipynb` writes epochs to `results/notebook.log`.

## Results

`results/fold_metrics.json` holds the five-fold metrics. `results/experiment.log` holds the epoch log. `results/figures/` holds the charts.

On the logged CUDA run, the two-stage model has overall MAE 0.772 ± 0.015. The paper reports 0.755 ± 0.005. The single network and the classical baselines are within about 0.02 of the published MAE, except Slope One (0.892 here, 0.916 in the paper).

Two protocol choices are not in the paper:

- Output biases start from Normal(0, 0.05), then move up by 1. Without that shift, a negative ReLU unit stays at 0.
- Top-10 metrics rank each test user against all 1,850 hotels. A user usually has one test hotel, so F1, MAP, and MRR stay near 0. The paper reports a much higher F1. It does not state the candidate set or the value of K.

## Tests

```bash
conda run -n rds python -m unittest tests.test_data tests.test_metrics
```

## Layout

| Path | Role |
| --- | --- |
| `src/data.py` | Read the archive, filter ratings, build folds |
| `src/models.py` | Criteria network, overall network, single network |
| `src/baselines.py` | Baseline, item-KNN, Slope One, SVD, SVD++ |
| `src/metrics.py` | MAE, F1, F2, FCP, MAP, MRR |
| `src/experiment.py` | Five-fold runner |
| `src/log.py` | Console log and file log |
