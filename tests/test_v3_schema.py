"""v3 feature lists, monotone constraints, and descriptions."""

from __future__ import annotations

import unittest

from src.schema import FEATURE_DESCRIPTIONS, MONOTONE_V3, V3_FEATURE_SETS, V3_PLAYER_FEATURES


class V3SchemaTests(unittest.TestCase):
    def test_every_v3_feature_has_a_description(self):
        missing = [n for n in V3_PLAYER_FEATURES if n not in FEATURE_DESCRIPTIONS]
        self.assertEqual(missing, [])

    def test_feature_lists_have_no_duplicates(self):
        self.assertEqual(len(V3_PLAYER_FEATURES), len(set(V3_PLAYER_FEATURES)))

    def test_monotone_keys_are_features_and_skip_opponent(self):
        self.assertTrue(set(MONOTONE_V3) <= set(V3_PLAYER_FEATURES))
        self.assertFalse(any(k.startswith("opp_") for k in MONOTONE_V3))

    def test_feature_sets_grow_and_end_at_all(self):
        sets = list(V3_FEATURE_SETS.values())
        for small, big in zip(sets, sets[1:]):
            self.assertTrue(set(small) < set(big))
        self.assertEqual(V3_FEATURE_SETS["all"], V3_PLAYER_FEATURES)
        self.assertEqual(len(V3_FEATURE_SETS["v2"]), 25)
