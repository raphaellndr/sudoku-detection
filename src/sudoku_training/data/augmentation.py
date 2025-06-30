"""
Data augmentation strategies for different datasets
"""

from keras import layers, models


class DataAugmentation:
    """Factory class for data augmentation strategies"""

    @staticmethod
    def mnist_augmentation() -> models.Sequential:
        """Create data augmentation for MNIST dataset"""
        return models.Sequential(
            [
                layers.RandomRotation(0.1),
                layers.RandomZoom(0.1),
                layers.RandomTranslation(0.1, 0.1),
            ]
        )

    @staticmethod
    def sudoku_augmentation() -> models.Sequential:
        """Create data augmentation for Sudoku dataset"""
        return models.Sequential(
            [
                layers.RandomRotation(0.1),
                layers.RandomZoom(0.1),
                layers.RandomTranslation(0.2, 0.2),
            ]
        )

    @staticmethod
    def custom_augmentation(
        rotation_factor: float = 0.1, zoom_factor: float = 0.1, translation_factor: float = 0.1
    ) -> models.Sequential:
        """Create custom data augmentation with specified parameters"""
        return models.Sequential(
            [
                layers.RandomRotation(rotation_factor),
                layers.RandomZoom(zoom_factor),
                layers.RandomTranslation(translation_factor, translation_factor),
            ]
        )
