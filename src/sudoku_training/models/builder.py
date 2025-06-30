"""
Model building utilities
"""

from keras import layers, models, regularizers

from sudoku_training.config import TrainingConfig


class ModelBuilder:
    """Builder class for creating CNN models"""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    def create_model(
        self, data_augmentation: models.Sequential, name: str = "model"
    ) -> models.Sequential:
        """Create a CNN model with specified data augmentation"""
        model = models.Sequential(
            [
                layers.Input(shape=(self.config.image_size, self.config.image_size, 1)),
                data_augmentation,
                layers.Conv2D(32, (3, 3), activation="relu"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation="relu"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(128, (3, 3), activation="relu"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Flatten(),
                layers.Dense(
                    256, activation="relu", kernel_regularizer=regularizers.l2(self.config.l2_reg)
                ),
                layers.Dropout(self.config.dropout_rate),
                layers.Dense(self.config.num_classes, activation="softmax"),
            ],
            name=name,
        )

        return model

    def create_lightweight_model(
        self, data_augmentation: models.Sequential, name: str = "lightweight_model"
    ) -> models.Sequential:
        """Create a lighter CNN model for faster training"""
        model = models.Sequential(
            [
                layers.Input(shape=(self.config.image_size, self.config.image_size, 1)),
                data_augmentation,
                layers.Conv2D(16, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(32, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Flatten(),
                layers.Dense(
                    128, activation="relu", kernel_regularizer=regularizers.l2(self.config.l2_reg)
                ),
                layers.Dropout(self.config.dropout_rate),
                layers.Dense(self.config.num_classes, activation="softmax"),
            ],
            name=name,
        )

        return model
