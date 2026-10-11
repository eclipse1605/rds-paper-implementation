"""Single-rating baselines: ALS biases, item-item KNN, Slope One, SVD, and SVD++.

Training uses only the arrays passed to each fitter. User and item ids that
never occur keep zero bias and zero factors, so a cold user scores as the
global mean plus the item term (Slope One falls back to the global mean).
"""

from __future__ import annotations

import numpy as np


def _training_arrays(
    user: np.ndarray, item: np.ndarray, rating: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    user = np.asarray(user, dtype=np.intp).reshape(-1)
    item = np.asarray(item, dtype=np.intp).reshape(-1)
    rating = np.asarray(rating, dtype=np.float64).reshape(-1)
    if user.shape != item.shape or user.shape != rating.shape:
        raise ValueError("user, item, and rating must have the same length")
    return user, item, rating


def _global_mean(rating: np.ndarray) -> float:
    if rating.size == 0:
        return 0.0
    return float(rating.mean())


def _last_observation(
    user: np.ndarray, item: np.ndarray, rating: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One row per (user, item). The last occurrence in the arrays wins."""

    if user.size == 0:
        return user, item, rating
    order = np.lexsort((np.arange(user.shape[0]), item, user))
    user = user[order]
    item = item[order]
    rating = rating[order]
    keep = np.empty(user.shape[0], dtype=bool)
    keep[-1] = True
    if user.shape[0] > 1:
        keep[:-1] = (user[:-1] != user[1:]) | (item[:-1] != item[1:])
    return user[keep], item[keep], rating[keep]


def _histories(
    user: np.ndarray, item: np.ndarray, value: np.ndarray
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    hist: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if user.size == 0:
        return hist
    order = np.argsort(user, kind="mergesort")
    user = user[order]
    item = item[order]
    value = value[order]
    cuts = np.flatnonzero(np.diff(user)) + 1
    starts = np.concatenate((np.array([0], dtype=np.intp), cuts))
    ends = np.concatenate((cuts, np.array([user.shape[0]], dtype=np.intp)))
    for start, end in zip(starts, ends):
        hist[int(user[start])] = (item[start:end].copy(), value[start:end].copy())
    return hist


def _seen_rows(ids: np.ndarray, n: int) -> np.ndarray:
    seen = np.zeros(n, dtype=bool)
    if ids.size:
        seen[np.unique(ids)] = True
    return seen


def _als_biases(
    user: np.ndarray,
    item: np.ndarray,
    rating: np.ndarray,
    n_users: int,
    n_items: int,
    n_epochs: int,
    reg_u: float,
    reg_i: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Alternating least squares. Each epoch updates every item, then every user."""

    mu = _global_mean(rating)
    bu = np.zeros(n_users, dtype=np.float64)
    bi = np.zeros(n_items, dtype=np.float64)
    if rating.size == 0 or n_epochs <= 0:
        return mu, bu, bi

    for _epoch in range(n_epochs):
        item_sum = np.bincount(item, weights=rating - mu - bu[user], minlength=n_items)
        item_count = np.bincount(item, minlength=n_items)
        bi[:] = 0.0
        seen_items = item_count > 0
        bi[seen_items] = item_sum[seen_items] / (reg_i + item_count[seen_items])

        user_sum = np.bincount(user, weights=rating - mu - bi[item], minlength=n_users)
        user_count = np.bincount(user, minlength=n_users)
        bu[:] = 0.0
        seen_users = user_count > 0
        bu[seen_users] = user_sum[seen_users] / (reg_u + user_count[seen_users])
    return mu, bu, bi


def _init_factors(
    n_rows: int, n_factors: int, seen: np.ndarray, rng: np.random.RandomState
) -> np.ndarray:
    factors = rng.normal(0.0, 0.1, size=(n_rows, n_factors)).astype(np.float64, copy=False)
    if n_rows:
        factors[~seen] = 0.0
    return factors


class _Predictor:
    """Shared pair scoring. `score_users` is the vectorized primitive."""

    n_items: int

    def predict_pairs(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        users = np.asarray(users, dtype=np.intp).reshape(-1)
        items = np.asarray(items, dtype=np.intp).reshape(-1)
        if users.shape[0] == 0:
            return np.empty(0, dtype=np.float64)
        unique_users, inverse = np.unique(users, return_inverse=True)
        scores = self.score_users(unique_users)
        return scores[inverse, items]

    def score_users(self, user_ids: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class BaselineModel(_Predictor):
    """Prediction is mu + bu + bi."""

    def __init__(self, mu: float, bu: np.ndarray, bi: np.ndarray):
        self.mu = float(mu)
        self.bu = bu
        self.bi = bi
        self.n_users = int(bu.shape[0])
        self.n_items = int(bi.shape[0])

    def score_users(self, user_ids: np.ndarray) -> np.ndarray:
        user_ids = np.asarray(user_ids, dtype=np.intp).reshape(-1)
        return self.mu + self.bu[user_ids, None] + self.bi[None, :]


class KNNBaselineModel(_Predictor):
    """Item-item interpolation of baseline residuals."""

    def __init__(
        self,
        mu: float,
        bu: np.ndarray,
        bi: np.ndarray,
        sim: np.ndarray,
        history: dict[int, tuple[np.ndarray, np.ndarray]],
        k: int,
    ):
        self.mu = float(mu)
        self.bu = bu
        self.bi = bi
        self.sim = sim
        self.history = history
        self.k = int(k)
        self.n_users = int(bu.shape[0])
        self.n_items = int(bi.shape[0])

    def score_users(self, user_ids: np.ndarray) -> np.ndarray:
        user_ids = np.asarray(user_ids, dtype=np.intp).reshape(-1)
        scores = self.mu + self.bu[user_ids, None] + self.bi[None, :]
        for row, user_id in enumerate(user_ids.tolist()):
            record = self.history.get(user_id)
            if record is None:
                continue
            neighbor_items, residuals = record
            _add_knn(scores[row], self.sim, neighbor_items, residuals, self.k)
        return scores


class SlopeOneModel(_Predictor):
    """Weighted Slope One. An item with no support on a user scores mu."""

    def __init__(
        self,
        mu: float,
        dev: np.ndarray,
        freq: np.ndarray,
        history: dict[int, tuple[np.ndarray, np.ndarray]],
    ):
        self.mu = float(mu)
        self.dev = dev
        self.freq = freq
        self.history = history
        self.n_items = int(dev.shape[0])

    def score_users(self, user_ids: np.ndarray) -> np.ndarray:
        user_ids = np.asarray(user_ids, dtype=np.intp).reshape(-1)
        scores = np.full((user_ids.shape[0], self.n_items), self.mu, dtype=np.float64)
        for row, user_id in enumerate(user_ids.tolist()):
            record = self.history.get(user_id)
            if record is None:
                continue
            items, ratings = record
            _slope_one_row(scores[row], self.dev, self.freq, items, ratings, self.mu)
        return scores


class SVDModel(_Predictor):
    """Biased SVD: mu + bu + bi + pu · qi."""

    def __init__(
        self,
        mu: float,
        bu: np.ndarray,
        bi: np.ndarray,
        pu: np.ndarray,
        qi: np.ndarray,
    ):
        self.mu = float(mu)
        self.bu = bu
        self.bi = bi
        self.pu = pu
        self.qi = qi
        self.n_users = int(bu.shape[0])
        self.n_items = int(bi.shape[0])
        self.n_factors = int(pu.shape[1])

    def score_users(self, user_ids: np.ndarray) -> np.ndarray:
        user_ids = np.asarray(user_ids, dtype=np.intp).reshape(-1)
        return self.mu + self.bu[user_ids, None] + self.bi[None, :] + self.pu[user_ids] @ self.qi.T


class SVDppModel(SVDModel):
    """SVD++. `pu` already includes the implicit term from the training set N(u)."""


def fit_baseline(
    user: np.ndarray,
    item: np.ndarray,
    rating: np.ndarray,
    n_users: int,
    n_items: int,
    n_epochs: int = 10,
    reg_u: float = 15.0,
    reg_i: float = 10.0,
) -> BaselineModel:
    user, item, rating = _training_arrays(user, item, rating)
    mu, bu, bi = _als_biases(user, item, rating, n_users, n_items, n_epochs, reg_u, reg_i)
    return BaselineModel(mu, bu, bi)


def fit_knn_baseline(
    user: np.ndarray,
    item: np.ndarray,
    rating: np.ndarray,
    n_users: int,
    n_items: int,
    k: int = 40,
) -> KNNBaselineModel:
    user, item, rating = _training_arrays(user, item, rating)
    mu, bu, bi = _als_biases(user, item, rating, n_users, n_items, n_epochs=10, reg_u=15.0, reg_i=10.0)
    unique_user, unique_item, unique_rating = _last_observation(user, item, rating)
    residuals = unique_rating - mu - bu[unique_user] - bi[unique_item]
    history = _histories(unique_user, unique_item, residuals)
    sim = _item_pearson(history, n_items)
    return KNNBaselineModel(mu, bu, bi, sim, history, k)


def fit_slope_one(
    user: np.ndarray,
    item: np.ndarray,
    rating: np.ndarray,
    n_users: int,
    n_items: int,
) -> SlopeOneModel:
    del n_users  # ids outside the training arrays are simply absent from history
    user, item, rating = _training_arrays(user, item, rating)
    mu = _global_mean(rating)
    unique_user, unique_item, unique_rating = _last_observation(user, item, rating)
    history = _histories(unique_user, unique_item, unique_rating)
    dev, freq = _slope_deviations(history, n_items)
    return SlopeOneModel(mu, dev, freq, history)


def fit_svd(
    user: np.ndarray,
    item: np.ndarray,
    rating: np.ndarray,
    n_users: int,
    n_items: int,
    n_factors: int = 100,
    n_epochs: int = 20,
    lr: float = 0.005,
    reg: float = 0.02,
    seed: int = 0,
) -> SVDModel:
    user, item, rating = _training_arrays(user, item, rating)
    mu = _global_mean(rating)
    bu = np.zeros(n_users, dtype=np.float64)
    bi = np.zeros(n_items, dtype=np.float64)
    rng = np.random.RandomState(seed)
    pu = _init_factors(n_users, n_factors, _seen_rows(user, n_users), rng)
    qi = _init_factors(n_items, n_factors, _seen_rows(item, n_items), rng)
    _sgd_svd(user, item, rating, mu, bu, bi, pu, qi, n_epochs, lr, reg)
    return SVDModel(mu, bu, bi, pu, qi)


def fit_svdpp(
    user: np.ndarray,
    item: np.ndarray,
    rating: np.ndarray,
    n_users: int,
    n_items: int,
    n_factors: int = 20,
    n_epochs: int = 20,
    lr: float = 0.007,
    reg: float = 0.02,
    seed: int = 0,
) -> SVDppModel:
    user, item, rating = _training_arrays(user, item, rating)
    mu = _global_mean(rating)
    bu = np.zeros(n_users, dtype=np.float64)
    bi = np.zeros(n_items, dtype=np.float64)
    unique_user, unique_item, unique_rating = _last_observation(user, item, rating)
    rated_items = {
        user_id: items for user_id, (items, _rating) in _histories(unique_user, unique_item, unique_rating).items()
    }
    rng = np.random.RandomState(seed)
    pu = _init_factors(n_users, n_factors, _seen_rows(user, n_users), rng)
    qi = _init_factors(n_items, n_factors, _seen_rows(item, n_items), rng)
    y = _init_factors(n_items, n_factors, _seen_rows(item, n_items), rng)
    _sgd_svdpp(user, item, rating, mu, bu, bi, pu, qi, y, rated_items, n_epochs, lr, reg)
    # N(u) is fixed after training, so fold |N(u)|^{-1/2} sum_j y_j into the user vector.
    effective = pu.copy()
    for user_id, items in rated_items.items():
        effective[user_id] += y[items].sum(axis=0) / np.sqrt(items.shape[0])
    return SVDppModel(mu, bu, bi, effective, qi)


def _item_pearson(
    history: dict[int, tuple[np.ndarray, np.ndarray]], n_items: int
) -> np.ndarray:
    """Pearson correlation of baseline residuals. Support below 2 stays zero.

    Residuals are already centered by the baseline, so the correlation is
    sum(e_i e_j) / sqrt(sum(e_i^2) sum(e_j^2)) over users who rated both.
    """

    sum_prod = np.zeros((n_items, n_items), dtype=np.float64)
    sum_sq = np.zeros((n_items, n_items), dtype=np.float64)
    count = np.zeros((n_items, n_items), dtype=np.float64)
    for items, residuals in history.values():
        if items.shape[0] < 2:
            continue
        sum_prod[np.ix_(items, items)] += np.outer(residuals, residuals)
        sum_sq[np.ix_(items, items)] += residuals[:, None] ** 2
        count[np.ix_(items, items)] += 1.0
    np.fill_diagonal(count, 0.0)
    np.fill_diagonal(sum_prod, 0.0)
    np.fill_diagonal(sum_sq, 0.0)
    denom = np.sqrt(np.maximum(sum_sq * sum_sq.T, 0.0))
    sim = np.zeros((n_items, n_items), dtype=np.float64)
    ok = (count >= 2.0) & (denom > 0.0)
    sim[ok] = sum_prod[ok] / denom[ok]
    np.fill_diagonal(sim, 0.0)
    return sim


def _add_knn(
    dest: np.ndarray,
    sim: np.ndarray,
    neighbor_items: np.ndarray,
    residuals: np.ndarray,
    k: int,
) -> None:
    """Add the k-neighbor residual interpolation in place. Denominator is sum |sim|."""

    n_neighbors = neighbor_items.shape[0]
    if n_neighbors == 0 or k <= 0:
        return
    if n_neighbors > k:
        selectable = sim[neighbor_items].copy()
        # The target item is not one of its own neighbors.
        selectable[np.arange(n_neighbors), neighbor_items] = -np.inf
        chosen = np.argpartition(selectable, n_neighbors - k, axis=0)[n_neighbors - k :]
        columns = np.arange(dest.shape[0])
        neighbor_sim = selectable[chosen, columns]
        neighbor_sim[~np.isfinite(neighbor_sim)] = 0.0
        residuals = residuals[chosen]
    else:
        neighbor_sim = sim[neighbor_items]
    if residuals.ndim == 1:
        numerator = (neighbor_sim * residuals[:, None]).sum(axis=0)
    else:
        numerator = (neighbor_sim * residuals).sum(axis=0)
    denominator = np.abs(neighbor_sim).sum(axis=0)
    supported = denominator > 0.0
    dest[supported] += numerator[supported] / denominator[supported]


def _slope_deviations(
    history: dict[int, tuple[np.ndarray, np.ndarray]], n_items: int
) -> tuple[np.ndarray, np.ndarray]:
    dev_sum = np.zeros((n_items, n_items), dtype=np.float64)
    freq = np.zeros((n_items, n_items), dtype=np.float64)
    for items, ratings in history.values():
        if items.shape[0] < 2:
            continue
        dev_sum[np.ix_(items, items)] += ratings[:, None] - ratings[None, :]
        freq[np.ix_(items, items)] += 1.0
    np.fill_diagonal(freq, 0.0)
    dev = np.zeros((n_items, n_items), dtype=np.float64)
    supported = freq > 0.0
    dev[supported] = dev_sum[supported] / freq[supported]
    return dev, freq


def _slope_one_row(
    dest: np.ndarray,
    dev: np.ndarray,
    freq: np.ndarray,
    items: np.ndarray,
    ratings: np.ndarray,
    mu: float,
) -> None:
    weight = freq[:, items]
    denominator = weight.sum(axis=1)
    supported = denominator > 0.0
    if not np.any(supported):
        dest[:] = mu
        return
    numerator = ((dev[:, items] + ratings[None, :]) * weight).sum(axis=1)
    dest[supported] = numerator[supported] / denominator[supported]
    dest[~supported] = mu


def _sgd_svd(
    user: np.ndarray,
    item: np.ndarray,
    rating: np.ndarray,
    mu: float,
    bu: np.ndarray,
    bi: np.ndarray,
    pu: np.ndarray,
    qi: np.ndarray,
    n_epochs: int,
    lr: float,
    reg: float,
) -> None:
    users = user.tolist()
    items = item.tolist()
    ratings = rating.tolist()
    lr = float(lr)
    reg = float(reg)
    for _epoch in range(n_epochs):
        for u, i, r in zip(users, items, ratings):
            p = pu[u].copy()
            q = qi[i].copy()
            err = r - (mu + bu[u] + bi[i] + float(p @ q))
            bu[u] += lr * (err - reg * bu[u])
            bi[i] += lr * (err - reg * bi[i])
            pu[u] = p + lr * (err * q - reg * p)
            qi[i] = q + lr * (err * p - reg * q)


def _sgd_svdpp(
    user: np.ndarray,
    item: np.ndarray,
    rating: np.ndarray,
    mu: float,
    bu: np.ndarray,
    bi: np.ndarray,
    pu: np.ndarray,
    qi: np.ndarray,
    y: np.ndarray,
    rated_items: dict[int, np.ndarray],
    n_epochs: int,
    lr: float,
    reg: float,
) -> None:
    users = user.tolist()
    items = item.tolist()
    ratings = rating.tolist()
    scale = {user_id: 1.0 / np.sqrt(items.shape[0]) for user_id, items in rated_items.items()}
    lr = float(lr)
    reg = float(reg)
    for _epoch in range(n_epochs):
        for u, i, r in zip(users, items, ratings):
            neighbors = rated_items[u]
            inv_sqrt = scale[u]
            implicit = y[neighbors].sum(axis=0) * inv_sqrt
            p = pu[u].copy()
            q = qi[i].copy()
            err = r - (mu + bu[u] + bi[i] + float(q @ (p + implicit)))
            bu[u] += lr * (err - reg * bu[u])
            bi[i] += lr * (err - reg * bi[i])
            pu[u] = p + lr * (err * q - reg * p)
            qi[i] = q + lr * (err * (p + implicit) - reg * q)
            # Advanced indexing returns a copy, so write the updated rows back.
            y[neighbors] = y[neighbors] + lr * (err * inv_sqrt * q - reg * y[neighbors])
