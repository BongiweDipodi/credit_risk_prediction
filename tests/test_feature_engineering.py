import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feature_engineering import encode_categorical_features


class TestFeatureEngineering(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "income": [50000, 70000, 50000],
                "occupation": ["Engineer", "Teacher", "Engineer"],
                "risk": ["low", "high", "low"],
            }
        )

    def test_one_hot_encoding(self):
        encoded_df, metadata = encode_categorical_features(
            self.df,
            categorical_columns=["occupation", "risk"],
            method="one_hot",
        )

        self.assertIn("occupation_Engineer", encoded_df.columns)
        self.assertIn("occupation_Teacher", encoded_df.columns)
        self.assertIn("risk_high", encoded_df.columns)
        self.assertNotIn("occupation", encoded_df.columns)
        self.assertEqual(metadata["method"], "one_hot")

    def test_label_encoding(self):
        encoded_df, metadata = encode_categorical_features(
            self.df,
            categorical_columns=["occupation"],
            method="label",
        )

        self.assertEqual(encoded_df["occupation"].tolist(), [0, 1, 0])
        self.assertIn("occupation", encoded_df.columns)
        self.assertEqual(metadata["method"], "label")


if __name__ == "__main__":
    unittest.main()
