"""
CLI commands for the training pipeline
"""

from pathlib import Path

import click

from sudoku_training.config import TrainingConfig
from sudoku_training.models.converter import convert_keras_model_to_onnx
from sudoku_training.training.pipeline import TrainingPipeline


@click.group()
def cli() -> None:
    """Sudoku Digit Recognition Training Pipeline."""


@cli.command()
@click.option("--data-dir", required=True, help="Path to Sudoku dataset directory")
@click.option("--pretrained-model", help="Path to pre-trained model")
@click.option("--use-mnist-pretraining", is_flag=True, help="Enable MNIST pre-training")
@click.option("--output-dir", default=".", help="Output directory for models")
@click.option("--batch-size", default=32, help="Batch size for training")
@click.option("--epochs", default=100, help="Number of epochs for Sudoku training")
@click.option("--learning-rate", default=0.001, help="Learning rate for training from scratch")
def train(
    data_dir,
    pretrained_model,
    use_mnist_pretraining,
    output_dir,
    batch_size,
    epochs,
    learning_rate,
) -> None:
    """Trains the Sudoku digit recognition model."""
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Create config
    config = TrainingConfig(batch_size=batch_size, finetune_epochs=epochs, scratch_lr=learning_rate)

    # Run pipeline
    pipeline = TrainingPipeline(config)
    results = pipeline.run(
        data_dir=data_dir,
        pretrained_model_path=pretrained_model,
        use_mnist_pretraining=use_mnist_pretraining,
        output_dir=output_dir,
    )

    # Print results
    click.echo("\n" + "=" * 50)
    click.echo("TRAINING COMPLETE")
    click.echo("=" * 50)
    click.echo(f"Training mode: {results['training_mode']}")
    click.echo(f"Final accuracy: {results['sudoku_accuracy']:.4f}")
    if "mnist_accuracy" in results:
        click.echo(f"MNIST accuracy: {results['mnist_accuracy']:.4f}")
    click.echo(f"Model saved to: {results['final_model_path']}")


@cli.command()
@click.option("--output-dir", default=".", help="Output directory for MNIST model")
@click.option("--epochs", default=50, help="Number of epochs for MNIST training")
def train_mnist(output_dir: str, epochs: int) -> None:
    """Trains only on MNIST dataset."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    config = TrainingConfig(mnist_epochs=epochs)
    pipeline = TrainingPipeline(config)

    results = pipeline.train_mnist_only(output_dir)

    click.echo(f"\nMNIST training complete! Accuracy: {results['mnist_accuracy']:.4f}")
    click.echo(f"Model saved to: {results['model_path']}")


@cli.command()
@click.option("--model-path", required=True, help="Path to trained model")
@click.option("--data-dir", required=True, help="Path to test dataset")
def evaluate(model_path: str, data_dir: str) -> None:
    """Evaluates a trained model."""
    config = TrainingConfig()
    pipeline = TrainingPipeline(config)

    try:
        results = pipeline.evaluate_model(model_path, data_dir)

        click.echo("\nEvaluation Results:")
        click.echo(f"Loss: {results['loss']:.4f}")
        click.echo(f"Accuracy: {results['accuracy']:.4f}")

    except FileNotFoundError as e:
        click.echo(f"Error: {e}")
        return


@cli.command()
@click.option("--model-path", required=True, help="Path to Keras model")
@click.option("--output-path", default=".", help="Output directory for ONNX model")
@click.option("--output-name", default=None, help="Optional name for the output ONNX model")
def convert(model_path: str, output_path: str, output_name: str = None) -> None:
    """Converts a Keras model to ONNX format."""
    model = Path(model_path)
    output_dir = Path(output_path)

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    convert_keras_model_to_onnx(model, output_dir, output_name)

    click.echo(
        f"Model converted to ONNX format and saved to: {output_dir / (output_name or model.stem + '.onnx')}"
    )


if __name__ == "__main__":
    cli()
