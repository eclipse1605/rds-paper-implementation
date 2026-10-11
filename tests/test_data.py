"""Checks for hotel parsing and the rating fold split."""

import unittest
from pathlib import Path

import numpy as np

from src.data import hotel_id_from_name, make_folds, parse_hotel_text, read_archive_member

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "texts.rar"
MEMBER = "Texts/Texts/hotel_100504_parsed.txt"


class HotelTextTests(unittest.TestCase):
    def test_hotel_100504_keeps_complete_ratings(self):
        text = read_archive_member(ARCHIVE, MEMBER)
        hotel_id = hotel_id_from_name(MEMBER)
        self.assertEqual(hotel_id, "100504")
        rows = parse_hotel_text(text, hotel_id)

        kept = [row for row in rows if row[0] == "ozzykat"]
        self.assertEqual(len(kept), 1)
        author, item, values = kept[0]
        self.assertEqual(author, "ozzykat")
        self.assertEqual(item, "100504")
        self.assertEqual(values, (5, 5, 5, 5, 5, 5, 5, 5))
        self.assertFalse(any(row[0] == "hothearted" for row in rows))

        for row_author, row_item, row_values in rows:
            self.assertEqual(row_item, "100504")
            self.assertEqual(len(row_values), 8)
            self.assertTrue(all(isinstance(value, int) and not isinstance(value, bool) for value in row_values))
            self.assertTrue(all(1 <= value <= 5 for value in row_values), row_author)


class FoldTests(unittest.TestCase):
    def test_make_folds_covers_each_index_once(self):
        n = 100
        folds = make_folds(n, n_folds=5, seed=0)
        self.assertEqual(len(folds), 5)
        test_counts = np.zeros(n, dtype=int)
        for fold in folds:
            fit = fold["fit"]
            val = fold["val"]
            test = fold["test"]
            self.assertEqual(len(val), 1)
            fit_set = set(int(index) for index in fit)
            val_set = set(int(index) for index in val)
            test_set = set(int(index) for index in test)
            self.assertTrue(fit_set.isdisjoint(val_set))
            self.assertTrue(fit_set.isdisjoint(test_set))
            self.assertTrue(val_set.isdisjoint(test_set))
            self.assertEqual(fit_set | val_set | test_set, set(range(n)))
            for index in test_set:
                test_counts[index] += 1
        self.assertTrue(np.all(test_counts == 1))


if __name__ == "__main__":
    unittest.main()
