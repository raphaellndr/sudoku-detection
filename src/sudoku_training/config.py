"""
Configuration module for training parameters
"""

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Configuration class for training parameters"""

    # Model architecture
    image_size: int = 28
    num_classes: int = 10

    # Training parameters
    batch_size: int = 32
    mnist_epochs: int = 50
    finetune_epochs: int = 100

    # Learning rates
    mnist_lr: float = 0.001
    finetune_lr: float = 0.00005
    scratch_lr: float = 0.001

    # Regularization
    l2_reg: float = 0.01
    dropout_rate: float = 0.5

    # Callbacks
    early_stopping_patience: int = 15
    lr_reduce_patience: int = 7
    min_lr: float = 1e-8

    @classmethod
    def from_dict(cls, config_dict: dict) -> "TrainingConfig":
        """Create config from dictionary"""
        return cls(**config_dict)

    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        return {
            field.name: getattr(self, field.name) for field in self.__dataclass_fields__.values()
        }
