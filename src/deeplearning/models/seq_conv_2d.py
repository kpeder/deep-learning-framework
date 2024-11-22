from contextlib2 import AbstractContextManager
from typing import Literal

import keras  # type: ignore
import keras_tuner as tuner  # type: ignore


class SequentialConv2D(keras.Sequential, AbstractContextManager):
    '''
    Sequential model using regularized convolutional neural network.
    '''
    def __init__(self, input_shape: tuple, num_classes: int):
        '''
        Implement the default model and return the instance.
        '''
        super().__init__([
            keras.layers.Input(shape=input_shape),
            keras.layers.Conv2D(32, kernel_size=(1, 1), activation="relu"),
            keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
            keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(num_classes, activation="softmax")])

    def __enter__(self):
        '''
        Context Manager entry method.
        '''
        return self

    def __exit__(self, *args) -> Literal[False]:
        '''
        Context Manager exit method.
        '''
        return False


class SequentialConv2DTunable(tuner.HyperModel, AbstractContextManager):
    '''
    A tunable Hypermodel using a sequential regularized convolutional neural network.
    '''
    def __init__(self, input_shape: tuple, num_classes: int, name: str = 'SeqConv2DTunable', metrics: list = [], verbose: int = 2):
        '''
        Initialize the class.
        '''
        super().__init__(name, tunable=True)

        self.input_shape: tuple = input_shape
        self.num_classes: int = num_classes
        self.metrics: list = metrics
        self.verbose: int = verbose

    def build(self, hp):
        '''
        Implement the model with Hyperparameters and return the instance.
        '''
        global_average_pool: bool = hp.Boolean('global_average_pool', True)
        initial_filters: int = hp.Int('initial_filters', 32, 64, step=32)
        initial_kernel_size: int = hp.Int('initial_kernel_size', 1, 3, step=2)
        loss_function = keras.losses.SparseCategoricalCrossentropy()
        optimizer_function = keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4]))

        model = keras.Sequential(name=self.name)

        model.add(keras.layers.Input(shape=self.input_shape)),
        model.add(keras.layers.Conv2D(initial_filters, kernel_size=(initial_kernel_size, initial_kernel_size), activation='relu')),
        model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu')),
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2))),
        model.add(keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu')),
        model.add(keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu')),
        if global_average_pool:
            model.add(keras.layers.GlobalAveragePooling2D()),
        else:
            model.add(keras.layers.MaxPooling2D(pool_size=(2, 2))),
            model.add(keras.layers.Flatten()),
        model.add(keras.layers.Dropout(hp.Float('dropout', 0.3, 0.7, step=0.2, sampling='linear'))),
        model.add(keras.layers.Dense(self.num_classes, activation='softmax'))

        model.compile(
            loss=loss_function,
            metrics=self.metrics,
            optimizer=optimizer_function
        )

        return model

    def fit(self,
            hp,
            model,
            x_train,
            y_train,
            callbacks: keras.callbacks.CallbackList):
        '''
        Train the model with Hyperparameters and return the instance.
        '''
        return model.fit(
            x_train,
            y_train,
            callbacks=callbacks,
            batch_size=hp['batch_size'],
            epochs=hp['epochs'],
            validation_split=hp['validation_split'],
            verbose=self.verbose
        )

    def __enter__(self):
        '''
        Context Manager entry method.
        '''
        return self

    def __exit__(self, *args) -> Literal[False]:
        '''
        Context Manager exit method.
        '''
        return False
