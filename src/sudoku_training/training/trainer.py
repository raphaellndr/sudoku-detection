"""
Model training utilities
"""

import click
import tensorflow as tf
from keras import models, optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sudoku_training.config import TrainingConfig
from sudoku_training.data.augmentation import DataAugmentation
from sudoku_training.models.builder import ModelBuilder


class Trainer:
    """Handles model training logic"""

    def __init__(self, config: TrainingConfig, model_builder: ModelBuilder) -> None:
        self.config = config
        self.model_builder = model_builder

    def get_callbacks(self, model_path: str, patience: int | None = None) -> list:
        """Create training callbacks"""
        patience = patience or self.config.early_stopping_patience

        return [
            EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
            ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=self.config.lr_reduce_patience,
                min_lr=self.config.min_lr,
            ),
        ]

    def train_mnist_model(
        self,
        x_train: tf.Tensor,
        y_train: tf.Tensor,
        x_test: tf.Tensor,
        y_test: tf.Tensor,
        save_path: str,
    ) -> tuple[models.Sequential, float]:
        """Train a model on MNIST dataset"""
        click.echo("Training on MNIST...")

        # Create model
        model = self.model_builder.create_model(
            DataAugmentation.mnist_augmentation(), "mnist_model"
        )

        # Compile
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.config.mnist_lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Train
        callbacks = self.get_callbacks(save_path, patience=10)

        model.fit(
            x_train,
            y_train,
            batch_size=self.config.batch_size,
            epochs=self.config.mnist_epochs,
            validation_data=(x_test, y_test),
            callbacks=callbacks,
            verbose=1,
        )

        # Evaluate
        _, accuracy = model.evaluate(x_test, y_test, verbose=0)
        click.echo(f"MNIST Test accuracy: {accuracy:.4f}")

        return model, accuracy

    def train_sudoku_model(
        self,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        pretrained_model: models.Sequential | None = None,
    ) -> tuple[models.Sequential, float]:
        """Train or fine-tune model on Sudoku dataset"""
        # Create model
        model = self.model_builder.create_model(
            DataAugmentation.sudoku_augmentation(), "sudoku_model"
        )

        # Transfer weights if available
        if pretrained_model is not None:
            self._transfer_weights(pretrained_model, model)
            lr = self.config.finetune_lr
            mode = "Fine-tuning"
        else:
            lr = self.config.scratch_lr
            mode = "Training from scratch"

        click.echo(f"Mode: {mode}")

        # Compile
        model.compile(
            optimizer=optimizers.Adam(learning_rate=lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Train
        callbacks = self.get_callbacks("sudoku_model.keras")

        model.fit(
            train_ds,
            epochs=self.config.finetune_epochs,
            validation_data=val_ds,
            callbacks=callbacks,
            verbose=1,
        )

        # Evaluate
        _, accuracy = model.evaluate(val_ds, verbose=0)

        return model, accuracy

    def _transfer_weights(
        self, source_model: models.Sequential, target_model: models.Sequential
    ) -> None:
        """Transfer weights from source to target model (excluding data augmentation)"""
        click.echo("Transferring weights from pre-trained model...")

        for i, layer in enumerate(target_model.layers):
            if i > 0:  # Skip data augmentation layer
                try:
                    if i < len(source_model.layers):
                        layer.set_weights(source_model.layers[i].get_weights())
                        click.echo(f"Transferred weights for layer {i}: {layer.name}")
                except Exception as e:
                    click.echo(f"Could not transfer weights for layer {i}: {layer.name} - {e}")


__all__ = ["Trainer"]
