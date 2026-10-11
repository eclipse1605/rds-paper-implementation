"""5-fold protocol for the two-stage multi-criteria DNN and the single-rating baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.data import CRITERIA, Ratings, load_ratings, make_folds
from src.log import configure, get_logger
from src.metrics import mae, ranking_metrics
from src.models import (
    LEARNING_RATE,
    OverallDNN,
    apply_zscore,
    criteria_dnn,
    evaluate_mae,
    predict,
    single_dnn,
    train_mae,
    zscore_stats,
)

METHOD_NAMES = ("ours", "single_dnn", "baseline", "knn", "slope_one", "svd", "svdpp")
TABLE6_ORDER = (
    ("svd", "SVD"),
    ("svdpp", "SVD++"),
    ("slope_one", "SlopeOne"),
    ("baseline", "Baseline estimates"),
    ("knn", "KNN with baseline"),
    ("single_dnn", "Single DNN"),
    ("ours", "Our"),
)
CRITERION_LABELS = {
    "value": "Value",
    "rooms": "Rooms",
    "location": "Location",
    "cleanliness": "Cleanliness",
    "checkin": "Check in / front desk",
    "service": "Service",
    "business": "Business service",
}
METRIC_KEYS = ("mae", "f1", "f2", "fcp", "map", "mrr")
RELEVANCE_THRESHOLD = 4.0
OVERFIT_ROWS = 48
OVERFIT_TRIES = 5
OVERFIT_MIN_DROP = 0.5
# A batch of 512 on a few dozen rows is one Adam step per epoch. At lr 0.001
# that does not move this ReLU stack by 0.5 inside 30 epochs, even when the
# output unit is alive. Cap the probe batch so those epochs contain enough steps.
OVERFIT_BATCH = 16
OVERFIT_MIN_EPOCHS = 30
PAIR_BATCH = 65536

# Surprise-style defaults from the experiment contract. Recorded in the results JSON.
BASELINE_HPARAMS = {
    "baseline": {"n_epochs": 10, "reg_u": 15.0, "reg_i": 10.0},
    "knn": {"k": 40},
    "slope_one": {},
    "svd": {"n_factors": 100, "n_epochs": 20, "lr": 0.005, "reg": 0.02},
    "svdpp": {"n_factors": 20, "n_epochs": 20, "lr": 0.007, "reg": 0.02},
}


def resolve_device(requested: str | None) -> str:
    if requested is None or requested == "":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="5-fold multi-criteria DNN experiment")
    parser.add_argument("--archive", default="texts.rar")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default=None,
        help="torch device (default: cuda if available, else cpu)",
    )
    parser.add_argument("--results", default="results/fold_metrics.json")
    parser.add_argument("--log", default="results/experiment.log")
    return parser.parse_args(argv)


def _embed_forward(model, batch):
    return model(batch[0], batch[1])


def _overall_forward(model, batch):
    return model(batch[0])


def _seed_torch(seed: int, device: str) -> None:
    torch.manual_seed(seed)
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _id_tensor(values: np.ndarray, device: str) -> torch.Tensor:
    return torch.as_tensor(np.asarray(values), dtype=torch.long, device=device)


def _float_tensor(values: np.ndarray, device: str) -> torch.Tensor:
    return torch.as_tensor(np.asarray(values), dtype=torch.float32, device=device)


def overfit_criteria(
    user: np.ndarray,
    item: np.ndarray,
    criteria: np.ndarray,
    n_users: int,
    n_items: int,
    *,
    device: str,
    max_epochs: int,
    patience: int,
    batch_size: int,
    base_seed: int,
    n_tries: int = OVERFIT_TRIES,
    min_drop: float = OVERFIT_MIN_DROP,
) -> dict[str, float | int]:
    """Memorize a few dozen rows. Abort when no seed drops train MAE by `min_drop`.

    train_mae always early-stops on a validation tuple. The same rows are passed
    as that tuple, so the watched value is train MAE and no held-out split is used.
    The step size is min(batch_size, 16, n_rows), and the probe runs at least 30
    epochs, so a live network can drop MAE by 0.5 before the fold budget matters.
    """

    if len(user) == 0:
        raise ValueError("overfit check needs at least one row")
    step_batch = max(1, min(int(batch_size), OVERFIT_BATCH, len(user)))
    probe_epochs = max(int(max_epochs), OVERFIT_MIN_EPOCHS)
    last: dict[str, float | int] | None = None
    for attempt in range(n_tries):
        seed = base_seed + attempt
        _seed_torch(seed, device)
        model = criteria_dnn(n_users, n_items)
        tensors = (
            _id_tensor(user, "cpu"),
            _id_tensor(item, "cpu"),
            _float_tensor(criteria, "cpu"),
        )
        model.to(device)
        moved = tuple(tensor.to(device) for tensor in tensors)
        initial = evaluate_mae(model, moved, _embed_forward)
        final = train_mae(
            model,
            tensors,
            tensors,
            _embed_forward,
            max_epochs=probe_epochs,
            patience=patience,
            batch_size=step_batch,
            learning_rate=LEARNING_RATE,
            seed=seed,
            device=device,
        )
        drop = float(initial - final)
        last = {
            "seed": seed,
            "initial_mae": float(initial),
            "final_mae": float(final),
            "drop": drop,
            "rows": int(len(user)),
            "batch_size": step_batch,
            "max_epochs": probe_epochs,
        }
        get_logger().info(
            "overfit seed %s: train MAE %.4f -> %.4f (drop %.4f)",
            seed,
            initial,
            final,
            drop,
        )
        if drop >= min_drop:
            return last
        del model
    raise SystemExit(
        "criteria_dnn overfit check failed: train MAE did not drop by "
        f"{min_drop} within {n_tries} seeds "
        f"(last seed {last['seed']}: {last['initial_mae']:.4f} -> {last['final_mae']:.4f})"
    )


def sample_overfit_rows(ratings: Ratings, n_rows: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    count = min(n_rows, len(ratings))
    chosen = rng.choice(len(ratings), size=count, replace=False)
    return ratings.user[chosen], ratings.item[chosen], ratings.criteria[chosen]


def _predict_embedding(model, users: np.ndarray, items: np.ndarray, device: str) -> np.ndarray:
    parts: list[np.ndarray] = []
    for start in range(0, len(users), PAIR_BATCH):
        stop = start + PAIR_BATCH
        pred = predict(
            model,
            (_id_tensor(users[start:stop], device), _id_tensor(items[start:stop], device)),
            _embed_forward,
        )
        parts.append(np.asarray(pred, dtype=np.float32))
    if not parts:
        out_dim = 1
        return np.zeros((0, out_dim), dtype=np.float32)
    return np.concatenate(parts, axis=0)


def _predict_overall(model, criteria: np.ndarray, device: str) -> np.ndarray:
    parts: list[np.ndarray] = []
    for start in range(0, len(criteria), PAIR_BATCH):
        stop = start + PAIR_BATCH
        pred = predict(
            model,
            (_float_tensor(criteria[start:stop], device),),
            _overall_forward,
        )
        parts.append(np.asarray(pred, dtype=np.float32).reshape(-1))
    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(parts, axis=0)


def _score_embedding(model, user_ids: np.ndarray, n_items: int, device: str) -> np.ndarray:
    """One forward covers every hotel for a chunk of users."""

    scores = np.empty((len(user_ids), n_items), dtype=np.float32)
    users_per = max(1, PAIR_BATCH // max(n_items, 1))
    item_index = torch.arange(n_items, device=device, dtype=torch.long)
    for start in range(0, len(user_ids), users_per):
        batch = np.asarray(user_ids[start : start + users_per], dtype=np.int64)
        width = len(batch)
        users = torch.as_tensor(batch, dtype=torch.long, device=device).repeat_interleave(n_items)
        items = item_index.repeat(width)
        pred = predict(model, (users, items), _embed_forward)
        scores[start : start + width] = np.asarray(pred, dtype=np.float32).reshape(width, n_items)
    return scores


def _score_ours(criteria_model, overall_model, user_ids, n_items, mean, std, device: str) -> np.ndarray:
    scores = np.empty((len(user_ids), n_items), dtype=np.float32)
    users_per = max(1, PAIR_BATCH // max(n_items, 1))
    item_index = torch.arange(n_items, device=device, dtype=torch.long)
    for start in range(0, len(user_ids), users_per):
        batch = np.asarray(user_ids[start : start + users_per], dtype=np.int64)
        width = len(batch)
        users = torch.as_tensor(batch, dtype=torch.long, device=device).repeat_interleave(n_items)
        items = item_index.repeat(width)
        criteria_pred = predict(criteria_model, (users, items), _embed_forward)
        z_pred = apply_zscore(np.asarray(criteria_pred, dtype=np.float32), mean, std)
        overall = _predict_overall(overall_model, z_pred, device)
        scores[start : start + width] = overall.reshape(width, n_items)
    return scores


def _seen_items(user: np.ndarray, item: np.ndarray) -> dict[int, np.ndarray]:
    if len(user) == 0:
        return {}
    order = np.argsort(user, kind="mergesort")
    users = np.asarray(user)[order]
    items = np.asarray(item)[order]
    breaks = np.flatnonzero(np.diff(users)) + 1
    starts = np.concatenate((np.array([0], dtype=breaks.dtype), breaks))
    ends = np.concatenate((breaks, np.array([len(users)], dtype=breaks.dtype)))
    seen: dict[int, np.ndarray] = {}
    for start, end in zip(starts, ends):
        seen[int(users[start])] = np.unique(items[start:end])
    return seen


def _pack_method(y_true: np.ndarray, y_pred: np.ndarray, ranking: dict[str, float]) -> dict[str, float]:
    packed = {"mae": mae(np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64).reshape(-1))}
    for key in ("f1", "f2", "fcp", "map", "mrr", "n_ranking_users"):
        packed[key] = float(ranking[key])
    return packed


def _score_ranking(test_user, test_item, test_overall, pred, scores, seen, k: int, n_items: int) -> dict[str, float]:
    unique_users = np.unique(test_user).astype(np.int64, copy=False)
    user_rows = {int(user): row for row, user in enumerate(unique_users)}
    if scores.shape != (len(unique_users), n_items):
        raise RuntimeError(
            f"score matrix shape {scores.shape} does not match {(len(unique_users), n_items)}"
        )
    ranking = ranking_metrics(
        test_user,
        test_item,
        test_overall,
        scores,
        user_rows,
        seen,
        k=k,
        relevance_threshold=RELEVANCE_THRESHOLD,
    )
    return _pack_method(test_overall, pred, ranking)


def _train_embedding(factory, users, items, target, val_users, val_items, val_target, *, seed, device, max_epochs, patience, batch_size):
    _seed_torch(seed, device)
    model = factory()
    train_mae(
        model,
        (_id_tensor(users, "cpu"), _id_tensor(items, "cpu"), _float_tensor(target, "cpu")),
        (_id_tensor(val_users, "cpu"), _id_tensor(val_items, "cpu"), _float_tensor(val_target, "cpu")),
        _embed_forward,
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        learning_rate=LEARNING_RATE,
        seed=seed,
        device=device,
    )
    return model


def _criteria_mae(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    per = {}
    values = []
    for index, name in enumerate(CRITERIA):
        value = mae(y_true[:, index], y_pred[:, index])
        per[name] = value
        values.append(value)
    per["mean"] = float(np.mean(values))
    return per


def _release(model, device: str) -> None:
    del model
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _fit_baselines():
    from src.baselines import (
        fit_baseline,
        fit_knn_baseline,
        fit_slope_one,
        fit_svd,
        fit_svdpp,
    )

    return {
        "baseline": fit_baseline,
        "knn": fit_knn_baseline,
        "slope_one": fit_slope_one,
        "svd": fit_svd,
        "svdpp": fit_svdpp,
    }


def run_fold(ratings: Ratings, fold: dict[str, np.ndarray], fold_index: int, *, device, max_epochs, patience, batch_size, top_k, seed) -> dict:
    fit = fold["fit"]
    val = fold["val"]
    test = fold["test"]
    train_index = np.concatenate((fit, val))
    test_user = ratings.user[test]
    test_item = ratings.item[test]
    test_overall = ratings.overall[test]
    seen = _seen_items(ratings.user[train_index], ratings.item[train_index])
    unique_users = np.unique(test_user).astype(np.int64, copy=False)
    fold_seed = seed + fold_index
    methods: dict[str, dict[str, float]] = {}

    get_logger().info("fold %s: criteria DNN", fold_index)
    criteria_model = _train_embedding(
        lambda: criteria_dnn(ratings.n_users, ratings.n_items),
        ratings.user[fit],
        ratings.item[fit],
        ratings.criteria[fit],
        ratings.user[val],
        ratings.item[val],
        ratings.criteria[val],
        seed=fold_seed,
        device=device,
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
    )
    criteria_pred = _predict_embedding(criteria_model, test_user, test_item, device)
    criteria_mae = _criteria_mae(ratings.criteria[test], criteria_pred)

    mean, std = zscore_stats(ratings.criteria[fit])
    get_logger().info("fold %s: overall DNN on true z-scored criteria", fold_index)
    _seed_torch(fold_seed, device)
    overall_model = OverallDNN()
    train_mae(
        overall_model,
        (
            _float_tensor(apply_zscore(ratings.criteria[fit], mean, std), "cpu"),
            _float_tensor(ratings.overall[fit][:, None], "cpu"),
        ),
        (
            _float_tensor(apply_zscore(ratings.criteria[val], mean, std), "cpu"),
            _float_tensor(ratings.overall[val][:, None], "cpu"),
        ),
        _overall_forward,
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        learning_rate=LEARNING_RATE,
        seed=fold_seed,
        device=device,
    )
    z_test = apply_zscore(criteria_pred, mean, std)
    ours_pred = _predict_overall(overall_model, z_test, device)
    ours_scores = _score_ours(criteria_model, overall_model, unique_users, ratings.n_items, mean, std, device)
    methods["ours"] = _score_ranking(
        test_user, test_item, test_overall, ours_pred, ours_scores, seen, top_k, ratings.n_items
    )
    _release(criteria_model, device)
    _release(overall_model, device)

    get_logger().info("fold %s: single DNN", fold_index)
    single_model = _train_embedding(
        lambda: single_dnn(ratings.n_users, ratings.n_items),
        ratings.user[fit],
        ratings.item[fit],
        ratings.overall[fit][:, None],
        ratings.user[val],
        ratings.item[val],
        ratings.overall[val][:, None],
        seed=fold_seed,
        device=device,
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
    )
    single_pred = _predict_embedding(single_model, test_user, test_item, device).reshape(-1)
    single_scores = _score_embedding(single_model, unique_users, ratings.n_items, device)
    methods["single_dnn"] = _score_ranking(
        test_user, test_item, test_overall, single_pred, single_scores, seen, top_k, ratings.n_items
    )
    _release(single_model, device)

    fitters = _fit_baselines()
    train_user = ratings.user[train_index]
    train_item = ratings.item[train_index]
    train_rating = ratings.overall[train_index]
    for name in ("baseline", "knn", "slope_one", "svd", "svdpp"):
        get_logger().info("fold %s: %s", fold_index, name)
        kwargs = dict(BASELINE_HPARAMS[name])
        if name in ("svd", "svdpp"):
            kwargs["seed"] = fold_seed
        model = fitters[name](
            train_user,
            train_item,
            train_rating,
            ratings.n_users,
            ratings.n_items,
            **kwargs,
        )
        pred = np.asarray(model.predict_pairs(test_user, test_item), dtype=np.float64).reshape(-1)
        scores = np.asarray(model.score_users(unique_users), dtype=np.float64)
        methods[name] = _score_ranking(
            test_user, test_item, test_overall, pred, scores, seen, top_k, ratings.n_items
        )
        del model

    return {"fold": fold_index, "criteria_mae": criteria_mae, "methods": methods}


def _mean_std(values: list[float]) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if len(array) > 1 else float("nan")
    return {"mean": mean, "std": std}


def summarize(folds: list[dict]) -> dict:
    criteria_summary = {}
    names = list(CRITERIA) + ["mean"]
    for name in names:
        criteria_summary[name] = _mean_std([fold["criteria_mae"][name] for fold in folds])
    method_summary = {}
    for method in METHOD_NAMES:
        method_summary[method] = {
            key: _mean_std([fold["methods"][method][key] for fold in folds])
            for key in METRIC_KEYS
        }
    return {"criteria_mae": criteria_summary, "methods": method_summary}


def _assumptions(args: argparse.Namespace, device: str) -> dict:
    return {
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "learning_rate": LEARNING_RATE,
        "relevance_threshold": RELEVANCE_THRESHOLD,
        "k": args.top_k,
        "fcp_tie_rule": (
            "Pairs with equal true ratings are ignored. Predicted ties count as neither "
            "concordant nor discordant. FCP is global nc / (nc + nd)."
        ),
        "neural_train_split": "fit only, early stopping on val, best validation weights restored",
        "baseline_train_split": "fit concatenated with val",
        "overall_dnn_training": (
            "Overall DNN trains and early-stops on z-scored true criteria "
            "(fit statistics). Test overall scores z-score the criteria network predictions "
            "with those same fit statistics."
        ),
        "seen_items": "items the user rated in fit or val",
        "zscore": "population mean and std of true fit criteria; std below 1e-8 becomes 1",
        "output_bias": (
            "Dense weights and biases are Normal(0, 0.05). Output biases are then "
            "shifted by +1 so ReLU(Wh+b) is active at initialization. Without that "
            "shift, negative output units stay at 0 and criteria MAE stays near 2.2."
        ),
        "std": "sample standard deviation with ddof=1 across folds",
        "baseline_hyperparameters": BASELINE_HPARAMS,
        "device": device,
        "overfit_rows": OVERFIT_ROWS,
        "overfit_min_drop": OVERFIT_MIN_DROP,
        "overfit_tries": OVERFIT_TRIES,
        "overfit_batch_size": min(args.batch_size, OVERFIT_BATCH, OVERFIT_ROWS),
        "overfit_max_epochs": max(args.max_epochs, OVERFIT_MIN_EPOCHS),
        "overfit_note": (
            "Probe batch is capped at 16 and the probe runs at least 30 epochs. "
            "Batch 512 on a few dozen rows is one update per epoch and does not "
            "drop train MAE by 0.5 within 30 epochs at learning rate 0.001, even "
            "when the output ReLU is alive."
        ),
    }


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not np.isfinite(number):
            return None
        return number
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    return value


def _format_pm(stats: dict) -> str:
    mean = stats["mean"]
    std = stats["std"]
    if std is None or not np.isfinite(std):
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {std:.4f}"


def log_tables(summary: dict) -> None:
    log = get_logger()
    log.info("Table 5  Criteria ratings error")
    log.info("%-24sMAE", "Criteria rating")
    for name in CRITERIA:
        log.info("%-24s%s", CRITERION_LABELS[name], _format_pm(summary["criteria_mae"][name]))
    log.info("%-24s%s", "Mean", _format_pm(summary["criteria_mae"]["mean"]))
    log.info("")
    log.info("Table 6  Evaluation results")
    labels = [label for _, label in TABLE6_ORDER]
    header = f"{'Metric':<8}" + "".join(f"{label:>22}" for label in labels)
    log.info("%s", header)
    for key in METRIC_KEYS:
        row = f"{key.upper():<8}"
        for method, _label in TABLE6_ORDER:
            row += f"{_format_pm(summary['methods'][method][key]):>22}"
        log.info("%s", row)


def run_experiment(ratings: Ratings, folds: list[dict[str, np.ndarray]], args: argparse.Namespace, device: str, overfit: dict) -> dict:
    fold_rows = [
        run_fold(
            ratings,
            fold,
            index,
            device=device,
            max_epochs=args.max_epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            top_k=args.top_k,
            seed=args.seed,
        )
        for index, fold in enumerate(folds)
    ]
    return {
        "archive": str(args.archive),
        "seed": args.seed,
        "n_ratings": len(ratings),
        "n_users": ratings.n_users,
        "n_items": ratings.n_items,
        "assumptions": _assumptions(args, device),
        "overfit": overfit,
        "folds": fold_rows,
        "summary": summarize(fold_rows),
    }


def write_results(payload: dict, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(_json_ready(payload), indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure(args.log)
    device = resolve_device(args.device)
    get_logger().info("device %s", device)
    ratings = load_ratings(args.archive)
    user, item, criteria = sample_overfit_rows(ratings, OVERFIT_ROWS, args.seed)
    overfit = overfit_criteria(
        user,
        item,
        criteria,
        ratings.n_users,
        ratings.n_items,
        device=device,
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        base_seed=args.seed,
    )
    folds = make_folds(len(ratings), n_folds=args.folds, seed=args.seed)
    payload = run_experiment(ratings, folds, args, device, overfit)
    write_results(payload, args.results)
    log_tables(payload["summary"])
    get_logger().info("wrote %s", args.results)


if __name__ == "__main__":
    main()
