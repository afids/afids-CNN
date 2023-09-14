from __future__ import annotations

import os
from argparse import ArgumentParser
from collections.abc import Iterable

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from tensorflow import keras

from afids_cnn.generator import customImageDataGenerator


def gen_training_array(
    num_channels: int,
    dims: NDArray,
    patches_path: os.PathLike[str] | str,
) -> NDArray:
    """Generate a training array containing patches of raw image and AFID location."""
    bps = 4 * num_channels * np.prod(dims)
    file_size = os.path.getsize(patches_path)
    num_samples = np.floor_divide(file_size, bps)

    arr_shape_train = (int(num_samples), dims[0], dims[1], dims[2], num_channels)

    arr_train = np.memmap(patches_path, "float32", "r", shape=arr_shape_train)
    return np.swapaxes(arr_train, 1, 3)


def create_generator(
    arr_train: NDArray,
    batch_size: int,
) -> Iterable[tuple[NDArray, NDArray]]:
    x_train = arr_train[..., 0]
    x_train = x_train.reshape(*x_train.shape[:4], 1)
    y_train = arr_train[..., 1]
    y_train = y_train.reshape(*y_train.shape[:4], 1)

    datagen_train = customImageDataGenerator()
    return datagen_train.flow(x_train, y_train, batch_size=batch_size)


def gen_conv3d_layer(
    filters: int,
    kernel_size: tuple[int, int, int] = (3, 3, 3),
) -> keras.layers.Conv3D:
    return keras.layers.Conv3D(filters, kernel_size, padding="same", activation="relu")


def gen_max_pooling_layer() -> keras.layers.MaxPooling3D:
    return keras.layers.MaxPooling3D((2, 2, 2))


def gen_transpose_layer(filters: int) -> keras.layers.Conv3DTranspose:
    return keras.layers.Conv3DTranspose(
        filters,
        kernel_size=2,
        strides=2,
        padding="same",
    )


def gen_std_block(filters: int, input_):
    x = gen_conv3d_layer(filters)(input_)
    out_layer = gen_conv3d_layer(filters)(x)
    return out_layer, gen_max_pooling_layer()(out_layer)


def gen_opposite_block(filters: int, input_, out_layer):
    x = input_
    for _ in range(3):
        x = gen_conv3d_layer(filters)(x)
    next_filters = filters // 2
    x = gen_transpose_layer(next_filters)(x)
    x = gen_conv3d_layer(next_filters)(x)
    return keras.layers.Concatenate(axis=4)([out_layer, x])


def gen_model() -> keras.Model:
    input_layer = keras.layers.Input((None, None, None, 1))
    x = keras.layers.ZeroPadding3D(padding=((1, 0), (1, 0), (1, 0)))(input_layer)

    out_layer_1, x = gen_std_block(16, x)  # block 1
    out_layer_2, x = gen_std_block(32, x)  # block 2
    out_layer_3, x = gen_std_block(64, x)  # block 3
    out_layer_4, x = gen_std_block(128, x)  # block 4

    # bottleneck
    x = gen_conv3d_layer(256)(x)
    x = gen_conv3d_layer(256)(x)
    x = keras.layers.Conv3DTranspose(filters=128, kernel_size=2, strides=(2, 2, 2))(x)
    x = gen_conv3d_layer(128, (2, 2, 2))(x)
    x = keras.layers.Concatenate(axis=4)([out_layer_4, x])

    x = gen_opposite_block(128, x, out_layer_3)  # block 5 (opposite 4)
    x = gen_opposite_block(64, x, out_layer_2)  # block 6 (opposite 3)
    x = gen_opposite_block(32, x, out_layer_1)  # block 7 (opposite 2)

    # block 8 (opposite 1)
    for _ in range(3):
        x = gen_conv3d_layer(16)(x)

    # output layer
    x = keras.layers.Cropping3D(cropping=((1, 0), (1, 0), (1, 0)), data_format=None)(x)
    x = keras.layers.Conv3D(1, (1, 1, 1), padding="same", activation=None)(x)

    return keras.Model(input_layer, x)


def fit_model(
    model: keras.Model,
    new_train: Iterable[tuple[NDArray, NDArray]],
    model_out_path: os.PathLike[str] | str,
    loss_out_path: os.PathLike[str] | str | None,
    epochs: int = 100,
    steps_per_epoch: int = 50,
    loss_fn: keras.losses.Loss | str = "mse",
    optimizer: keras.optimizers.Optimizer | str | None = None,
    metrics: list[keras.metrics.Metric | str] | None = None,
    validation_data: Iterable[tuple[NDArray, NDArray]] | None = None,
    validation_steps: int = 50,
    callbacks: Iterable[keras.callbacks.Callback] | None = None,
):
    if not optimizer:
        optimizer = keras.optimizers.Adam()
    if not metrics:
        metrics = [keras.metrics.RootMeanSquaredError()]

    model.compile(
        loss=[loss_fn],
        optimizer=optimizer,
        metrics=metrics,
    )
    history = model.fit(
        new_train,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_data,
        validation_steps=validation_steps,
        callbacks=callbacks,
    )
    model.save(model_out_path)
    if loss_out_path:
        pd.DataFrame(history.history).to_csv(loss_out_path)
    return history, model


def gen_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("num_channels", type=int)
    parser.add_argument("radius", type=int)
    parser.add_argument("patches_path")
    parser.add_argument("model_out_path")
    parser.add_argument("--loss_out_path")
    parser.add_argument("--validation_patches_path")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps_per_epoch", type=int, default=50)
    parser.add_argument("--loss_fn", default="mse")
    parser.add_argument("--optimizer", default="adam")
    parser.add_argument("--metrics", nargs="*", default=["RootMeanSquaredError"])
    parser.add_argument("--validation_steps", type=int, default=50)
    parser.add_argument("--do_early_stopping", action="store_true")
    return parser


def main():
    args = gen_parser().parse_args()
    model = gen_model()
    new_train = create_generator(
        gen_training_array(
            args.num_channels,
            np.array([(args.radius * 2) + 1 for _ in range(3)]),
            args.patches_path,
        ),
        batch_size=10,
    )
    validation_data = (
        create_generator(
            gen_training_array(
                args.num_channels,
                np.array([(args.radius * 2) + 1 for _ in range(3)]),
                args.validation_patches_path,
            ),
            batch_size=10,
        )
        if args.validation_patches_path
        else None
    )

    callbacks = (
        [keras.callbacks.EarlyStopping(monitor="val_loss", patience=100)]
        if args.do_early_stopping
        else None
    )
    fit_model(
        model,
        new_train,
        args.model_out_path,
        args.loss_out_path,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        loss_fn=args.loss_fn,
        optimizer=args.optimizer,
        metrics=args.metrics,
        validation_data=validation_data,
        validation_steps=args.validation_steps,
        callbacks=callbacks,
    )


if __name__ == "__main__":
    main()
