import h5py
import numpy as np
from tensorflow.keras.utils import Sequence

from .utils import create_frames_array, labes_to_onehot


class HacsGenerator(Sequence):

    def __init__(self, file_path, data_keys, nb_frames=16, frame_shape=(112, 112, 3),
                 batch_size=16, use_negative_samples=False, shuffle=False):
        self._file_path = file_path
        self._labels_key = data_keys['labels']
        self._frames_key = data_keys['frames']
        self._classes_key = data_keys['classes']
        self._nb_frames = nb_frames
        self._frame_shape = frame_shape
        self._use_negative_samples = use_negative_samples
        self._batch_size = batch_size
        self._indices = self._get_indices()

        print(f"\n\nProcessing: {file_path}")
        if shuffle:
            print("Shuffling indices")
            np.random.shuffle(self._indices)
        print(f"First 10 indices: {self._indices[:10]}\n\n")

    def _get_indices(self):
        with h5py.File(self._file_path, 'r') as file_handle:
            labels_handle = file_handle[self._labels_key]
            if self._use_negative_samples:
                indices = np.arange(len(labels_handle))
            else:
                indices = np.where(labels_handle[:] == 1)[0]

            return indices

    def __len__(self):
        return int(np.floor(len(self._indices) / self._batch_size))

    def __getitem__(self, index):
        selected_indices = self._indices[index * self._batch_size:(index + 1) * self._batch_size]
        selected_indices = sorted(selected_indices)

        return self._generate_data(selected_indices)

    def _generate_data(self, indices):
        with h5py.File(self._file_path, 'r') as file_handle:
            classes = file_handle[self._classes_key][indices]
            videos = file_handle[self._frames_key][indices]

            frames = create_frames_array(videos, nb_frames=self._nb_frames,
                                         frame_shape=self._frame_shape)

            if self._use_negative_samples:
                labels = file_handle[self._labels_key][indices]
                one_hot_labels = labes_to_onehot(labels)
                return frames, [classes, one_hot_labels]

            return frames, classes


class HacsGeneratorPartial(HacsGenerator):
    """
    Generates data in smaller parts. It allows for dividing training into smaller "epochs".
    """

    def __init__(self, file_path, data_keys, nb_frames=16, frame_shape=(112, 112, 3),
                 batch_size=16, use_negative_samples=False, shuffle=False, samples_per_part=10000):
        super().__init__(file_path, data_keys, nb_frames, frame_shape, batch_size, use_negative_samples, shuffle)

        self._temp_indices = self._indices[:samples_per_part]
        self._samples_per_part = samples_per_part
        self._initial_step = 0

    def __len__(self):
        return int(np.floor(len(self._temp_indices) / self._batch_size))

    def __getitem__(self, index):
        selected_indices = self._temp_indices[index * self._batch_size:
                                              (index + 1) * self._batch_size]
        selected_indices = sorted(selected_indices)

        return self._generate_data(selected_indices)

    def get_epoch_multiplier(self):
        """
        Single epoch refers to iterating through all training samples.
        To mimic that with the partial generator number of epochs must be
        multiplied accordingly.

        If there are 10000 samples in a training set and we use the partial
        generator with 1000 samples per part, to iterate through the whole
        data set once we have to train model for 10 epochs.

        :return: Integer by which number of epochs must be multiplied
           to iterate through all samples in data set.
        """

        return int(np.floor(len(self._indices) / self._samples_per_part))

    def on_epoch_end(self):
        new_start = (self._initial_step + 1) * self._samples_per_part
        self._initial_step += 1

        if new_start >= len(self._indices):
            new_start = 0
            self._initial_step = 0

        print(f"\nProcessing {new_start}:{ new_start+self._samples_per_part} part of the data")
        self._temp_indices = self._indices[new_start: new_start+self._samples_per_part]
