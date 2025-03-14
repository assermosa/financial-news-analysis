import logging

def setup_logging():
    """
    Sets up logging configuration.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )

def save_model(model, filepath):
    """
    Saves the trained model to a file.
    """
    model.save(filepath)
    logging.info(f"Model saved to {filepath}")