from pathlib import Path
from typing import Any

import tensorflow as tf
from keras.models import Sequential, load_model
from tf2onnx.convert import from_keras


def convert_keras_model_to_onnx(
    model: Path | Sequential,
    output_path: Path,
    output_name: str | None = None,
) -> Any:
    """Converts a keras model to ONNX format and saves it to the specified path.

    Args:
        model (Path | Sequential): The Keras model to convert (or its path).
        output_path (Path): The path where the ONNX model will be saved.
        output_name (str | None): Optional name for the output model file. If None, uses model's name.

    Returns:
        Any: model_proto.
    """
    if isinstance(model, Path):
        keras_model = load_model(model)
    else:
        keras_model = model

    # Set output names if not already set
    if not hasattr(keras_model, "output_names") or keras_model.output_names is None:
        keras_model.output_names = ["output"]

    # Create output path
    if output_name is not None:
        output_file_path = output_path / f"{output_name}.onnx"
    else:
        model_name = getattr(keras_model, "name", "model")
        output_file_path = output_path / f"{model_name}.onnx"

    # Get the input shape from the model
    if hasattr(keras_model, "input_shape") and keras_model.input_shape is not None:
        input_shape = keras_model.input_shape
        print(f"Model input shape: {input_shape}")
    else:
        raise ValueError("Cannot determine model input shape")

    spec = tf.TensorSpec(
        shape=input_shape,
        dtype=tf.float32,
        name="input",
    )

    try:
        model_proto, _ = from_keras(
            keras_model,
            input_signature=[spec],
            opset=13,
            output_path=str(output_file_path),
        )

        print(f"ONNX conversion successful! Output file: {output_file_path}")

        return model_proto

    except Exception as e:
        print(f"ONNX conversion failed: {e}")
        raise
