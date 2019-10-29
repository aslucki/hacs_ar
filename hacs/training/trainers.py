import os

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad
from keras_radam.training import RAdamOptimizer

from hacs.training.data_generator.hacs_generator import HacsGeneratorPartial


class C3DTrainer:
    def __init__(self, model, use_labels, data_file_keys, output_dir, learning_rate):
        self._model = model
        self._use_labels = use_labels
        self._data_file_keys = data_file_keys
        self._output_dir = output_dir
        self._initial_epoch = 0
        self._learning_rate = learning_rate
        self._optimizer = None

    def _maybe_load_checkpoint(self):
        model_files = [file for file in os.listdir(self._output_dir)
                       if 'model' in file and file.split('.')[-1] == 'h5']

        if len(model_files) > 0:
            model_files.sort(key=lambda x: int(x.split('_')[1]))
            model_path = os.path.join(self._output_dir, model_files[-1])
            self._model = load_model(model_path)
            print(f'Loading from checkpoint: {model_path}')
            initial_epoch = int(model_files[-1].split('_')[1])
            self._initial_epoch = initial_epoch

        else:
            print(f'No checkpoints, training from start')

    def compile_model(self, optimizer, losses, metrics):
        self._maybe_load_checkpoint()

        if optimizer.lower() == 'radam':
            optimizer = RAdamOptimizer(total_steps=1000, warmup_proportion=0.1,
                                       learning_rate=self._learning_rate,
                                       min_lr=1e-8)
        elif optimizer.lower() == 'adam':
            optimizer = Adam(lr=self._learning_rate)
        elif optimizer.lower() == 'adagrad':
            optimizer = Adagrad(lr=self._learning_rate)
        elif optimizer.lower() == 'sgd':
            optimizer = SGD(lr=self._learning_rate, momentum=0.9)
        elif optimizer.lower() == 'rmsprop':
            optimizer = RMSprop(lr=self._learning_rate)

        self._optimizer = optimizer

        return self._model.compile(optimizer=self._optimizer, loss=losses, metrics=metrics)

    def get_generator(self, file_handle, batch_size=24, validation=False):
        samples_per_part = 10000
        if validation:
            samples_per_part = 2000

        generator = HacsGeneratorPartial(file_handle,  self._data_file_keys,
                                         use_negative_samples=self._use_labels,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         samples_per_part=samples_per_part)
        return generator


    @staticmethod
    def _maybe_multiply_epochs(generator, epochs):
        try:
            multiplier = generator.get_epoch_multiplier()
        except AttributeError:
            multiplier = 1
        return epochs * multiplier

    def train(self, train_file_handle, validation_file_handle, epochs=10, batch_size=24):
        file_path = os.path.join(self._output_dir, 'model_{epoch:02d}_{val_loss:.2f}.h5')
        checkpoint_callback = ModelCheckpoint(file_path, monitor='val_loss', save_best_only=True)
        early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0, patience=50,
                                                mode='auto')

        tensorboard_callback = TensorBoard(log_dir=os.path.join(self._output_dir, 'logs'), histogram_freq=1,
                                           batch_size=batch_size,
                                           write_grads=True, write_images=True,
                                           update_freq='epoch')

        train_gen = self.get_generator(train_file_handle, batch_size)
        validation_gen = self.get_generator(validation_file_handle, batch_size, validation=True)

        epochs = self._maybe_multiply_epochs(train_gen, epochs)

        print(f'Will train for {epochs} epochs with batch size equal to {batch_size}')
        print(f'Will process {len(train_gen)} steps per epoch')
        print(f'Will process {len(validation_gen)} validation steps')
        print(self._model.loss)
        print(self._model.summary())

        history = self._model.fit_generator(generator=train_gen,
                                            epochs=epochs,
                                            initial_epoch=self._initial_epoch,
                                            validation_data=validation_gen,
                                            use_multiprocessing=True,
                                            workers=6,
                                            callbacks=[checkpoint_callback, early_stopping_callback,
                                                       tensorboard_callback])

        return history
