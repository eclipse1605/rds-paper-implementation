"""TripAdvisor hotel ratings from the author's Texts.rar archive."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CRITERIA = (
    "value",
    "rooms",
    "location",
    "cleanliness",
    "checkin",
    "service",
    "business",
)
RATING_COLUMNS = ("overall",) + CRITERIA
# <Rating> order in the parsed hotel files: overall, then the seven criteria.
N_RATING_FIELDS = 8
_BSDTAR_FALLBACK = "/home/advaitam/miniconda3/bin/bsdtar"


def bsdtar_executable() -> str:
    """Locate bsdtar. conda run does not put the base env bin on PATH."""

    found = shutil.which("bsdtar")
    if found:
        return found
    return _BSDTAR_FALLBACK


@dataclass
class Ratings:
    """Integer-coded ratings. One row is one complete review."""

    user: np.ndarray
    item: np.ndarray
    overall: np.ndarray
    criteria: np.ndarray
    n_users: int
    n_items: int

    def __len__(self) -> int:
        return int(self.user.shape[0])


def parse_hotel_text(text: str, hotel_id: str) -> list[tuple]:
    """Return complete (author, hotel_id, 8 ratings) rows.

    A row is kept only when all eight ratings are integers in 1..5.
    Missing marks such as -1 drop the row.
    """

    rows: list[tuple] = []
    author: str | None = None
    for line in text.splitlines():
        if line.startswith("<Author>"):
            author = line[len("<Author>") :].strip()
        elif line.startswith("<Rating>") and author:
            parts = line[len("<Rating>") :].split()
            if len(parts) < N_RATING_FIELDS:
                continue
            try:
                values = tuple(int(part) for part in parts[:N_RATING_FIELDS])
            except ValueError:
                continue
            if all(1 <= value <= 5 for value in values):
                rows.append((author, hotel_id, values))
    return rows


def hotel_id_from_name(path: str) -> str:
    name = Path(path).name
    prefix = "hotel_"
    suffix = "_parsed.txt"
    if name.startswith(prefix) and name.endswith(suffix):
        return name[len(prefix) : -len(suffix)]
    raise ValueError(f"unexpected hotel filename: {path}")


def load_ratings(archive: str | Path) -> Ratings:
    """Stream hotel files out of the rar archive and encode user and hotel ids."""

    archive = Path(archive)
    raw_user: list[str] = []
    raw_item: list[str] = []
    raw_values: list[tuple] = []
    with tempfile.TemporaryDirectory(prefix="tripadvisor-") as tmp:
        subprocess.run(
            [bsdtar_executable(), "-xf", str(archive), "-C", tmp],
            check=True,
        )
        root = Path(tmp)
        paths = sorted(root.rglob("hotel_*_parsed.txt"))
        if not paths:
            raise FileNotFoundError(f"no hotel files in {archive}")
        for path in paths:
            hotel_id = hotel_id_from_name(path.name)
            text = path.read_text(encoding="utf-8", errors="replace")
            for author, item, values in parse_hotel_text(text, hotel_id):
                raw_user.append(author)
                raw_item.append(item)
                raw_values.append(values)

    user_labels = {label: index for index, label in enumerate(sorted(set(raw_user)))}
    item_labels = {label: index for index, label in enumerate(sorted(set(raw_item)))}
    values = np.asarray(raw_values, dtype=np.float32)
    return Ratings(
        user=np.fromiter((user_labels[label] for label in raw_user), dtype=np.int32, count=len(raw_user)),
        item=np.fromiter((item_labels[label] for label in raw_item), dtype=np.int32, count=len(raw_item)),
        overall=values[:, 0],
        criteria=values[:, 1:],
        n_users=len(user_labels),
        n_items=len(item_labels),
    )


def make_folds(n: int, n_folds: int = 5, seed: int = 42) -> list[dict[str, np.ndarray]]:
    """Random rating splits: 20% test, 1% of all ratings as validation, the rest fit.

    The validation slice is taken from the non-test ratings, matching
    "80% training set including a 1% validation set".
    """

    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    blocks = np.array_split(order, n_folds)
    val_size = int(round(0.01 * n))
    if val_size < 1:
        raise ValueError("dataset is too small for a 1% validation split")
    folds = []
    for fold_id, test in enumerate(blocks):
        rest = np.concatenate([block for i, block in enumerate(blocks) if i != fold_id])
        if val_size >= len(rest):
            raise ValueError("validation split is larger than the training pool")
        folds.append(
            {
                "fit": np.sort(rest[val_size:]),
                "val": np.sort(rest[:val_size]),
                "test": np.sort(test),
            }
        )
    return folds


def read_archive_member(archive: str | Path, member: str) -> str:
    completed = subprocess.run(
        [bsdtar_executable(), "-xOf", str(archive), member],
        check=True,
        capture_output=True,
    )
    return completed.stdout.decode("utf-8", errors="replace")
