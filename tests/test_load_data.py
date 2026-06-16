import pandas as pd
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processing import load_data


class TestLoadData(unittest.TestCase):
    """Test cases for load_data function"""
    
    @classmethod
    def setUpClass(cls):
        """Create test data directory"""
        cls.test_dir = Path("tests/fixtures")
        cls.test_dir.mkdir(parents=True, exist_ok=True)
    
    def setUp(self):
        """Create sample test file before each test"""
        sample_data = {
            'age': [25, 35, 45, 55],
            'income': [50000, 75000, 100000, 150000],
            'loan_amount': [10000, 25000, 50000, 100000],
            'employment_status': ['Employed', 'Employed', 'Employed', 'Self-Employed'],
            'credit_score': [680, 720, 750, 780]
        }
        self.sample_df = pd.DataFrame(sample_data)
        self.test_file = self.test_dir / "sample_data.csv"
        self.sample_df.to_csv(self.test_file, index=False)
    
    def tearDown(self):
        """Clean up test files"""
        if self.test_file.exists():
            self.test_file.unlink()
    
    def test_load_valid_csv(self):
        """Test loading a valid CSV file"""
        df = load_data(str(self.test_file))
        
        self.assertEqual(df.shape, (4, 5))
        self.assertEqual(len(df.columns), 5)
        self.assertListEqual(list(df.columns), 
                           ['age', 'income', 'loan_amount', 'employment_status', 'credit_score'])
    
    def test_load_nonexistent_file(self):
        """Test loading a non-existent file raises FileNotFoundError"""
        with self.assertRaises(FileNotFoundError):
            load_data("nonexistent_file.csv")
    
    def test_load_empty_file(self):
        """Test loading an empty CSV file raises ValueError"""
        empty_file = self.test_dir / "empty.csv"
        empty_file.write_text("")
        
        try:
            with self.assertRaises(ValueError):
                load_data(str(empty_file))
        finally:
            empty_file.unlink()
    
    def test_load_noncsv_file(self):
        """Test loading a non-CSV file logs warning but attempts load"""
        non_csv_file = self.test_dir / "sample.txt"
        non_csv_file.write_text("age,income\n25,50000\n")
        
        try:
            df = load_data(str(non_csv_file))
            self.assertIsNotNone(df)
        finally:
            non_csv_file.unlink()
    
    def test_load_corrupted_csv(self):
        """Test loading a corrupted CSV raises ValueError"""
        corrupted_file = self.test_dir / "corrupted.csv"
        corrupted_file.write_text("this is\nnot valid csv data [[[\n")
        
        try:
            with self.assertRaises(ValueError):
                load_data(str(corrupted_file))
        finally:
            corrupted_file.unlink()


if __name__ == "__main__":
    unittest.main()
