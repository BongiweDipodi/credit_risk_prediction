import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.train_model import build_training_dataset, split_data, train_model_pipeline


class TestTrainModel(unittest.TestCase):
    def test_split_data(self):
        df = pd.DataFrame(
            {
                "income": [100, 200, 300, 400],
                "risk": [0, 0, 1, 1],
            }
        )
        train_features, test_features, train_target, test_target = split_data(df, target_column="risk")

        self.assertEqual(train_features.shape[0] + test_features.shape[0], len(df))
        self.assertEqual(len(train_target), len(train_features))
        self.assertEqual(len(test_target), len(test_features))

    def test_build_training_dataset(self):
        df = pd.DataFrame(
            {
                "income": [100, 200, 300, 400],
                "risk": [0, 0, 1, 1],
            }
        )
        dataset = build_training_dataset(df, target_column="risk")

        self.assertIn("train_features", dataset)
        self.assertIn("test_features", dataset)
        self.assertIn("train_target", dataset)
        self.assertIn("test_target", dataset)
        self.assertEqual(dataset["target_column"], "risk")

    def test_train_model_pipeline(self):
        df = pd.DataFrame(
            {
                "income": [100, 200, 300, 400, 500, 600],
                "loan_amount": [10, 20, 30, 40, 50, 60],
                "risk": [0, 0, 0, 1, 1, 1],
            }
        )
        result = train_model_pipeline(df, target_column="risk")

        self.assertIn("model", result)
        self.assertIn("metrics", result)
        self.assertIn("accuracy", result["metrics"])
        self.assertIn("precision", result["metrics"])
        self.assertIn("recall", result["metrics"])
        self.assertIn("f1", result["metrics"])
        self.assertIn("cv_mean_accuracy", result["metrics"])
        self.assertIn("cv_std_accuracy", result["metrics"])

    def test_train_model_pipeline_persists_artifacts(self):
        df = pd.DataFrame(
            {
                "income": [100, 200, 300, 400, 500, 600],
                "loan_amount": [10, 20, 30, 40, 50, 60],
                "risk": [0, 0, 0, 1, 1, 1],
            }
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = train_model_pipeline(df, target_column="risk", persist_artifacts=True, artifact_dir=tmp_dir)

            self.assertTrue(Path(tmp_dir, "random_forest_model.joblib").exists())
            self.assertTrue(Path(tmp_dir, "training_metrics.json").exists())
            self.assertIn("artifact_paths", result)


if __name__ == "__main__":
    unittest.main()
