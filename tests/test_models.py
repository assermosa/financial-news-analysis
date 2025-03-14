import unittest
import numpy as np
import tensorflow as tf
from src.models.train import build_model
from src.models.evaluate import evaluate_model
from src.models.inference import predict

class TestModels(unittest.TestCase):
    def test_build_model(self):
        # Test if the model is built correctly
        model = build_model()
        self.assertIsInstance(model, tf.keras.Model)

    def test_evaluate_model(self):
        # Test model evaluation
        model = build_model()
        X_test = np.random.rand(10, 128)  # Random input data
        test_mask = np.random.rand(10, 128)  # Random attention mask
        y_test_sent = np.random.randint(0, 2, size=(10,))  # Random sentiment labels
        y_test_rank = np.random.rand(10, 1)  # Random rank labels

        eval_results = evaluate_model(model, X_test, test_mask, y_test_sent, y_test_rank)
        self.assertIsInstance(eval_results, list)  # Ensure evaluation returns a list of metrics

    def test_predict(self):
        # Test inference
        model = build_model()
        input_text = "This is a test sentence."
        tokenizer = tf.keras.preprocessing.text.Tokenizer()
        tokenizer.fit_on_texts([input_text])
        sentiment_pred, rank_pred = predict(model, input_text, tokenizer)
        self.assertIsInstance(sentiment_pred, np.ndarray)  # Ensure sentiment prediction is returned
        self.assertIsInstance(rank_pred, np.ndarray)  # Ensure rank prediction is returned

if __name__ == "__main__":
    unittest.main()