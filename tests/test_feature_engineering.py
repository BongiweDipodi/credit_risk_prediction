import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feature_engineering import (
    analyze_feature_importance,
    build_feature_pipeline,
    encode_categorical_features,
    scale_numeric_features,
    select_features,
)


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

    def test_standard_scaler(self):
        scaled_df, metadata = scale_numeric_features(
            pd.DataFrame({"income": [100, 200, 300], "loan_amount": [10, 20, 30]}),
            numeric_columns=["income", "loan_amount"],
        )

        self.assertAlmostEqual(float(scaled_df["income"].mean()), 0.0, places=6)
        self.assertAlmostEqual(float(scaled_df["income"].std(ddof=0)), 1.0, places=6)
        self.assertEqual(metadata["scaled_columns"], ["income", "loan_amount"])

    def test_feature_selection(self):
        df = pd.DataFrame(
            {
                "income": [100, 200, 300, 400],
                "loan_amount": [20, 30, 40, 50],
                "risk": [0, 0, 1, 1],
            }
        )
        selected_df, metadata = select_features(df, target_column="risk", top_k=2)

        self.assertIn("risk", selected_df.columns)
        self.assertEqual(len(metadata["selected_features"]), 2)
        self.assertTrue(all(col in selected_df.columns for col in metadata["selected_features"]))

    def test_feature_pipeline(self):
        df = pd.DataFrame(
            {
                "income": [100, 200, 300, 400],
                "occupation": ["Engineer", "Teacher", "Engineer", "Teacher"],
                "risk": [0, 0, 1, 1],
            }
        )
        transformed_df, metadata = build_feature_pipeline(
            df,
            target_column="risk",
            categorical_columns=["occupation"],
            numeric_columns=["income"],
            top_k=2,
        )

        self.assertIn("risk", transformed_df.columns)
        self.assertIn("encoding", metadata)
        self.assertIn("scaling", metadata)
        self.assertIn("selection", metadata)

    def test_feature_importance_analysis(self):
        features = pd.DataFrame({"income": [100, 200, 300, 400], "loan_amount": [10, 20, 30, 40]})
        target = pd.Series([0, 0, 1, 1])
        metadata = analyze_feature_importance(features, target, top_k=2)

        self.assertEqual(metadata["model"], "RandomForestClassifier")
        self.assertEqual(len(metadata["top_features"]), 2)
        self.assertTrue(all(feature in metadata["importance_scores"] for feature in metadata["top_features"]))


if __name__ == "__main__":
    unittest.main()
