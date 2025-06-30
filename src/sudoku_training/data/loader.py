"""
Data loading and preprocessing utilities
"""

import tensorflow as tf
from keras.datasets import mnist
from keras.preprocessing import image_dataset_from_directory
from keras.utils import to_categorical

from sudoku_training.config import TrainingConfig


class DataLoader:
    """Handles data loading and preprocessing"""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    def load_mnist_data(self) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Load and preprocess MNIST dataset"""
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # Reshape and normalize
        x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
        x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

        # Convert to categorical
        y_train = to_categorical(y_train, self.config.num_classes)
        y_test = to_categorical(y_test, self.config.num_classes)

        return x_train, y_train, x_test, y_test

    def load_sudoku_dataset(self, data_dir: str) -> tuple[tf.data.Dataset, tf.data.Dataset]:
        """Load and preprocess Sudoku dataset"""
        train_ds = image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(self.config.image_size, self.config.image_size),
            batch_size=self.config.batch_size,
            label_mode="categorical",
        )

        val_ds = image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(self.config.image_size, self.config.image_size),
            batch_size=self.config.batch_size,
            label_mode="categorical",
        )

        # Apply preprocessing
        train_ds = train_ds.map(self._preprocess_sudoku).prefetch(buffer_size=tf.data.AUTOTUNE)
        val_ds = val_ds.map(self._preprocess_sudoku).prefetch(buffer_size=tf.data.AUTOTUNE)

        return train_ds, val_ds

    def _preprocess_sudoku(self, image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """Preprocess Sudoku images"""
        image = tf.image.rgb_to_grayscale(image)
        image = tf.cast(image, tf.float32) / 255.0
        return image, label
