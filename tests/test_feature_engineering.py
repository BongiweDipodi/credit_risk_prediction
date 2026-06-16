import pandas as pd
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feature_engineering import encode_categorical, scale_numerical, select_features


class TestFeatureEngineering(unittest.TestCase):
    """Tests for categorical encoding functions."""

    @classmethod
    def setUpClass(cls):
        cls.test_data = pd.DataFrame({
            'age': [25, 35, 45, 55],
            'income': [50000, 75000, 100000, 150000],
            'employment_status': ['Employed', 'Employed', 'Self-Employed', 'Unemployed'],
            'credit_history': ['Good', 'Fair', 'Good', 'Poor'],
        })

    def test_label_encoding(self):
        encoded_df, encoders = encode_categorical(
            self.test_data,
            categorical_columns=['employment_status', 'credit_history'],
            method='label',
        )

        self.assertEqual(encoded_df.shape[1], 4)
        self.assertIn('employment_status', encoders)
        self.assertIn('credit_history', encoders)
        self.assertTrue(all(encoded_df['employment_status'].isin([0, 1, 2])))
        self.assertTrue(all(encoded_df['credit_history'].isin([0, 1, 2])))

    def test_onehot_encoding(self):
        encoded_df, encoder = encode_categorical(
            self.test_data,
            categorical_columns=['employment_status', 'credit_history'],
            method='onehot',
        )

        self.assertGreater(encoded_df.shape[1], 4)
        self.assertEqual(encoded_df.index.tolist(), self.test_data.index.tolist())
        self.assertTrue(all(col.startswith('employment_status_') or col.startswith('credit_history_') or col in ['age', 'income'] for col in encoded_df.columns))
        self.assertEqual(len(encoder.categories_), 2)

    def test_scale_numerical_defaults(self):
        scaled_df, scaler = scale_numerical(self.test_data, numerical_columns=['age', 'income'])

        self.assertEqual(scaled_df.shape, self.test_data.shape)
        self.assertTrue(hasattr(scaler, 'mean_'))
        self.assertAlmostEqual(scaled_df['age'].mean(), 0.0, places=6)
        self.assertAlmostEqual(scaled_df['income'].mean(), 0.0, places=6)
        self.assertAlmostEqual(scaled_df['age'].std(ddof=0), 1.0, places=6)
        self.assertAlmostEqual(scaled_df['income'].std(ddof=0), 1.0, places=6)

    def test_scale_numerical_auto_select(self):
        scaled_df, scaler = scale_numerical(self.test_data)

        self.assertIn('age', scaled_df.columns)
        self.assertIn('income', scaled_df.columns)
        self.assertTrue(all(col in scaler.feature_names_in_ for col in ['age', 'income']))

    def test_select_features_f_classif(self):
        target = pd.Series([0, 1, 0, 1])
        selected_df, selected_cols = select_features(
            self.test_data[['age', 'income']],
            target,
            method='f_classif',
            k=1,
        )

        self.assertEqual(selected_df.shape[1], 1)
        self.assertEqual(len(selected_cols), 1)
        self.assertIn(selected_cols[0], ['age', 'income'])

    def test_select_features_model_importance(self):
        target = pd.Series([0, 1, 0, 1])
        selected_df, selected_cols = select_features(
            self.test_data[['age', 'income']],
            target,
            method='model_importance',
            k=2,
        )

        self.assertEqual(selected_df.shape[1], 2)
        self.assertEqual(len(selected_cols), 2)

    def test_select_features_invalid_method(self):
        target = pd.Series([0, 1, 0, 1])
        with self.assertRaises(ValueError):
            select_features(self.test_data[['age', 'income']], target, method='invalid')

    def test_select_features_k_all(self):
        target = pd.Series([0, 1, 0, 1])
        selected_df, selected_cols = select_features(
            self.test_data[['age', 'income']],
            target,
            method='f_classif',
            k='all',
        )

        self.assertEqual(len(selected_cols), 2)
        self.assertEqual(selected_df.shape[1], 2)

    def test_select_features_mismatched_lengths(self):
        target = pd.Series([0, 1, 0])
        with self.assertRaises(ValueError):
            select_features(self.test_data[['age', 'income']], target, method='f_classif')

    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            encode_categorical(
                self.test_data,
                categorical_columns=['employment_status', 'credit_history'],
                method='unsupported',
            )


if __name__ == '__main__':
    unittest.main()
