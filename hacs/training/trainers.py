import os

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import load_model
from keras_radam.training import RAdamOptimizer

from .data_generators import data_generator, data_generator_with_shuffle, data_generator_labels


class C3DTrainer:
    def __init__(self, model, use_labels, data_file_keys, output_dir, initial_learning_rate):
        self._model = model
        self._use_labels = use_labels
        self._data_file_keys = data_file_keys
        self._output_dir = output_dir
        self._initial_epoch = 0
        self._initial_learning_rate = initial_learning_rate

    def _lr_schedule(self, epoch):
        if epoch == 0:
            lr = self._initial_learning_rate
        else:
            lr = self._initial_learning_rate/epoch
        return lr

    def _maybe_load_checkpoint(self):
        model_files = [file for file in os.listdir(self._output_dir)
                       if 'model' in file and file.split('.')[-1] == 'h5']

        if len(model_files) > 0:
            model_files = sorted(model_files)
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
            optimizer = RAdamOptimizer(total_steps=5000, warmup_proportion=0.1, min_lr=1e-5)
        return self._model.compile(optimizer=optimizer, loss=losses, metrics=metrics)

    def calculate_nb_steps(self, file_handle, batch_size, samples_to_cache=1000):
        if self._use_labels:
            lens = []
            for k in file_handle.keys():
                lens.append(len(file_handle[k]))
            nb_samples = max(lens)

        else:
            nb_samples = 0
            for i in range(0, len(file_handle[self._data_file_keys['labels']]), samples_to_cache):
                temp = file_handle[self._data_file_keys['labels']][i:i + samples_to_cache]
                nb_samples += len(temp[temp == 1])

        return int(nb_samples/batch_size)

    def get_generator(self, file_handle, batch_size=24):
        generator = data_generator_labels(file_handle, self._data_file_keys,
                                          yield_labels=self._use_labels,
                                          batch_size=batch_size)
        return generator

    def train(self, train_file_handle, validation_file_handle, epochs=10, batch_size=24):
        file_path = os.path.join(self._output_dir, 'model_{epoch:02d}_{val_loss:.2f}.h5')
        checkpoint_callback = ModelCheckpoint(file_path, monitor='val_loss', save_best_only=True)
        early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0, patience=10,
                                                mode='auto')
        tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1, batch_size=batch_size,
                                           write_grads=True, write_images=True,
                                           update_freq='epoch')

        train_gen = self.get_generator(train_file_handle, batch_size)
        validation_gen = self.get_generator(validation_file_handle, batch_size)

        steps_per_epoch = int(self.calculate_nb_steps(train_file_handle, batch_size)/40)
        validation_steps = int(self.calculate_nb_steps(validation_file_handle, batch_size)/40)

        print(f'Will train for {epochs} epochs with batch size equal to {batch_size}')
        print(f'Will process {steps_per_epoch} steps per epoch')
        print(f'Will process {validation_steps} validation steps')
        print(self._model.loss)
        history = self._model.fit_generator(generator=train_gen,
                                            steps_per_epoch=steps_per_epoch,
                                            epochs=epochs,
                                            initial_epoch=self._initial_epoch,
                                            validation_data=validation_gen,
                                            validation_steps=validation_steps,
                                            callbacks=[checkpoint_callback, early_stopping_callback,
                                                       tensorboard_callback])

        return history


