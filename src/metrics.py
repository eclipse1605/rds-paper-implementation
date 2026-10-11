"""MAE and top-K ranking metrics used in section 4.2."""

from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def f_beta(precision: float, recall: float, beta: float) -> float:
    beta_sq = beta * beta
    denominator = beta_sq * precision + recall
    if denominator == 0.0:
        return 0.0
    return (1.0 + beta_sq) * precision * recall / denominator


def _average(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(values))


def ranking_metrics(
    test_user: np.ndarray,
    test_item: np.ndarray,
    test_rating: np.ndarray,
    scores: np.ndarray,
    user_rows: dict[int, int],
    seen_items: dict[int, np.ndarray],
    *,
    k: int = 10,
    relevance_threshold: float = 4.0,
) -> dict[str, float]:
    """Macro F1, F2, MAP, MRR and global FCP.

    Relevance is a true rating at or above `relevance_threshold`. Users with
    no such test item are left out of precision, recall, F1, F2, MAP, and MRR.
    FCP follows Koren: pairs with unequal true ratings are concordant when the
    predicted order agrees and discordant when it disagrees. Predicted ties
    count as neither.
    """

    precisions: list[float] = []
    recalls: list[float] = []
    f1_values: list[float] = []
    f2_values: list[float] = []
    average_precisions: list[float] = []
    reciprocal_ranks: list[float] = []
    concordant = 0
    discordant = 0

    order = np.argsort(test_user, kind="mergesort")
    users = test_user[order]
    items = test_item[order]
    ratings = test_rating[order]
    breaks = np.flatnonzero(np.diff(users)) + 1
    groups = np.split(np.arange(len(users)), breaks)

    for group in groups:
        user = int(users[group[0]])
        row = user_rows[user]
        user_scores = scores[row]
        group_items = items[group]
        group_ratings = ratings[group]
        # Average duplicate user-item test rows so each hotel has one label.
        unique_items, inverse = np.unique(group_items, return_inverse=True)
        summed = np.zeros(len(unique_items), dtype=np.float64)
        counts = np.zeros(len(unique_items), dtype=np.float64)
        np.add.at(summed, inverse, group_ratings)
        np.add.at(counts, inverse, 1.0)
        label = summed / counts
        predicted = user_scores[unique_items]

        concordant, discordant = _accumulate_fcp(label, predicted, concordant, discordant)

        relevant = unique_items[label >= relevance_threshold]
        if len(relevant) == 0:
            continue
        relevant_set = set(int(item) for item in relevant)
        masked = user_scores.copy()
        seen = seen_items.get(user)
        if seen is not None and len(seen):
            masked[seen] = -np.inf
        finite = np.isfinite(masked)
        if not np.any(finite):
            continue
        ranked = np.argsort(-masked, kind="mergesort")
        ranked = ranked[np.isfinite(masked[ranked])]
        top = ranked[:k]
        n_recommended = int(len(top))
        if n_recommended == 0:
            continue
        hits = sum(1 for item in top if int(item) in relevant_set)
        precision = hits / n_recommended
        recall = hits / len(relevant_set)
        precisions.append(precision)
        recalls.append(recall)
        f1_values.append(f_beta(precision, recall, 1.0))
        f2_values.append(f_beta(precision, recall, 2.0))
        average_precisions.append(_average_precision(top, relevant_set, k, len(relevant_set)))
        reciprocal_ranks.append(_reciprocal_rank(top, relevant_set))

    pair_total = concordant + discordant
    return {
        "f1": _average(f1_values),
        "f2": _average(f2_values),
        "fcp": float(concordant / pair_total) if pair_total else float("nan"),
        "map": _average(average_precisions),
        "mrr": _average(reciprocal_ranks),
        "n_ranking_users": float(len(precisions)),
    }


def _accumulate_fcp(
    label: np.ndarray,
    predicted: np.ndarray,
    concordant: int,
    discordant: int,
) -> tuple[int, int]:
    n = len(label)
    for left in range(n):
        for right in range(left + 1, n):
            true_gap = float(label[left] - label[right])
            if true_gap == 0.0:
                continue
            pred_gap = float(predicted[left] - predicted[right])
            if pred_gap == 0.0:
                continue
            if true_gap * pred_gap > 0.0:
                concordant += 1
            else:
                discordant += 1
    return concordant, discordant


def _average_precision(top: np.ndarray, relevant: set[int], k: int, n_relevant: int) -> float:
    hits = 0
    total = 0.0
    for rank, item in enumerate(top, start=1):
        if int(item) in relevant:
            hits += 1
            total += hits / rank
    return total / min(k, n_relevant)


def _reciprocal_rank(top: np.ndarray, relevant: set[int]) -> float:
    for rank, item in enumerate(top, start=1):
        if int(item) in relevant:
            return 1.0 / rank
    return 0.0
