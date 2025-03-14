import unittest
import pandas as pd
from src.data_processing.preprocess import clean_text
from src.data_processing.clean_data import clean_data

class TestDataProcessing(unittest.TestCase):
    def test_clean_text(self):
        # Test text cleaning
        input_text = "Hello! This is a test. http://example.com"
        expected_output = "hello this is a test"
        self.assertEqual(clean_text(input_text), expected_output)

    def test_clean_data(self):
        # Test data cleaning
        data = pd.DataFrame({
            "news": ["Sample news 1", "Sample news 2", None],
            "sentiment": ["POSITIVE", "NEGATIVE", "POSITIVE"]
        })
        cleaned_data = clean_data(data)
        self.assertEqual(len(cleaned_data), 2)  # Ensure rows with missing values are dropped

if __name__ == "__main__":
    unittest.main()