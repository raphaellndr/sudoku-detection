"""
Training pipeline orchestrator
"""

from pathlib import Path

import click
from keras.models import load_model

from sudoku_training.config import TrainingConfig
from sudoku_training.data.loader import DataLoader
from sudoku_training.models.builder import ModelBuilder

from .trainer import Trainer


class TrainingPipeline:
    """Main training pipeline orchestrator"""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.model_builder = ModelBuilder(config)
        self.data_loader = DataLoader(config)
        self.trainer = Trainer(config, self.model_builder)

    def run(
        self,
        data_dir: str,
        pretrained_model_path: str | None = None,
        output_dir: str = ".",
        *,
        use_mnist_pretraining: bool = False,
    ) -> dict:
        """Run the complete training pipeline"""
        results = {}
        pretrained_model = None

        # Handle pre-training
        if pretrained_model_path and Path(pretrained_model_path).exists():
            click.echo(f"Loading pre-trained model from: {pretrained_model_path}")
            pretrained_model = load_model(pretrained_model_path)

        elif use_mnist_pretraining:
            click.echo("Pre-training on MNIST...")
            x_train, y_train, x_test, y_test = self.data_loader.load_mnist_data()
            mnist_save_path = str(Path(output_dir) / "mnist_pretrained.keras")
            pretrained_model, mnist_acc = self.trainer.train_mnist_model(
                x_train, y_train, x_test, y_test, mnist_save_path
            )
            results["mnist_accuracy"] = mnist_acc

        # Load Sudoku dataset
        click.echo("Loading Sudoku dataset...")
        train_ds, val_ds = self.data_loader.load_sudoku_dataset(data_dir)

        # Train on Sudoku
        click.echo("Training on Sudoku dataset...")
        sudoku_model, sudoku_acc = self.trainer.train_sudoku_model(
            train_ds, val_ds, pretrained_model
        )

        # Save final model
        final_model_path = str(Path(output_dir) / "sudoku_digits_recognition.keras")
        sudoku_model.save(final_model_path)

        results.update(
            {
                "sudoku_accuracy": sudoku_acc,
                "final_model_path": final_model_path,
                "training_mode": "fine-tuning" if pretrained_model else "from_scratch",
            }
        )

        return results

    def train_mnist_only(self, output_dir: str = ".") -> dict:
        """Train only on MNIST dataset"""
        click.echo("Training MNIST model...")

        # Load data and train
        x_train, y_train, x_test, y_test = self.data_loader.load_mnist_data()
        save_path = str(Path(output_dir) / "mnist_model.keras")

        model, accuracy = self.trainer.train_mnist_model(
            x_train, y_train, x_test, y_test, save_path
        )

        return {"mnist_accuracy": accuracy, "model_path": save_path}

    def evaluate_model(self, model_path: str, data_dir: str) -> dict:
        """Evaluate a trained model"""
        if not Path(model_path).exists():
            error_msg = f"Model file {model_path} does not found!"
            raise FileNotFoundError(error_msg)

        # Load model
        model = load_model(model_path)

        # Load test data
        _, val_ds = self.data_loader.load_sudoku_dataset(data_dir)

        # Evaluate
        loss, accuracy = model.evaluate(val_ds, verbose=1)

        return {"loss": loss, "accuracy": accuracy}
