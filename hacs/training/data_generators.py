from itertools import groupby, count
import traceback

import cv2
import numpy as np

from hacs.processing.frame_extractor import decode_frames


def find_last_frame(frames):
    """
    Some 2s videos have fewer than 60 frames due to the frame rate.
    Frames arrays are padded with frames containing ones in all channels.
    The functions finds the first frame where all pixel values equal one.

    :param frames: Array of frames (4-dimensional)
    :return: Index of the first padded frame
    """
    summed_frames = np.sum(frames, axis=(1, 2, 3))
    sum_to_find = np.product(frames.shape[1:])
    indexes = np.where(summed_frames == sum_to_find)[0]

    if len(indexes) > 0:
        c = count()
        val = max((list(g) for _, g in groupby(indexes, lambda x: x - next(c))),
                  key=len)

        return val[0]

    return len(frames)


def crop_frames(frames, size=(112, 112)):
    """
    Randomly crops frames to the required size.
    Serves as data augmentation.

    :param frames: Array of frames (4-dimensional)
    :param size: Size of cropped frames
    :return: Array of cropped frames
    """

    frames_size = frames.shape[-3:-1]

    try:
        start_y = np.random.randint(0, frames_size[0] - size[0], size=1)[0]
        start_x = np.random.randint(0, frames_size[1] - size[1], size=1)[0]

    except ValueError:
        start_y = 0
        start_x = 0

    cropped = []
    for frame in frames:
        cropped.append(frame[start_y:start_y + size[0],
                       start_x:start_x + size[1], :])

    return np.asarray(cropped)


def resize_frames(frames, size=(112, 112)):

    resized = []
    for frame in frames:
        resized.append(cv2.resize(frame, size))

    return np.asarray(resized)


def labes_to_onehot(labels):
    one_hot = np.zeros(shape=(len(labels), 2))
    one_hot[labels == 1, 1] = 1
    one_hot[labels == -1, 0] = 1

    return one_hot.astype(np.float32)


def convert_to_floats(frames, max_value=255.):
    return frames/max_value


def create_frames_array(videos, nb_frames):
    frames = []
    for video in videos:
        decoded = decode_frames(video)
        last_frame = find_last_frame(decoded)

        start_frame = 0
        if last_frame > nb_frames:
            start_frame = np.random.randint(0, last_frame - nb_frames, size=1)[0]

        cropped = crop_frames(decoded[start_frame:start_frame + nb_frames])
        floating = convert_to_floats(cropped)
        frames.append(floating)

    return np.asarray(frames)


def data_generator(file_handle, data_file_keys, yield_labels=False, batch_size=1, nb_frames=16):
    index = 0
    data_len = len(file_handle[data_file_keys['classes']])
    while True:
        try:
            classes = file_handle[data_file_keys['classes']][index:index + batch_size]
            classes = classes.astype(np.float32)
            videos = file_handle[data_file_keys['frames']][index:index + batch_size]
            frames = create_frames_array(videos, nb_frames=nb_frames)

            if yield_labels:
                labels = file_handle[data_file_keys['labels']][index:index + batch_size]
                one_hot = labes_to_onehot(labels)
                index += batch_size
                yield np.asarray(frames), [classes, one_hot]

            index += batch_size
            if index > data_len:
                index = 0
            yield np.asarray(frames), classes

        except ValueError:
            traceback.print_exc()


def data_generator_with_shuffle(file_handle, data_file_keys, yield_labels=False, batch_size=1, nb_frames=16,
                                batch_to_shuffle=1000):
    index = 0
    data_len = len(file_handle[data_file_keys['classes']])
    while True:
        try:
            indices = np.arange(batch_to_shuffle, dtype=int)
            np.random.shuffle(indices)

            classes = file_handle[data_file_keys['classes']][index:index + batch_to_shuffle][indices]
            classes = classes.astype(np.float32)
            videos = file_handle[data_file_keys['frames']][index:index + batch_to_shuffle][indices]

            if yield_labels:
                labels = file_handle[data_file_keys['labels']][index:index + batch_to_shuffle][indices]

            for i in range(0, batch_to_shuffle, batch_size):
                batch_classes = classes[i:i + batch_size]
                batch_videos = videos[i:i + batch_size]
                frames = create_frames_array(batch_videos, nb_frames=nb_frames)

                if yield_labels:
                    batch_labels = labels[i:i + batch_size]
                    one_hot = labes_to_onehot(batch_labels)
                    yield frames, [batch_classes, one_hot]

                yield frames, batch_classes

            index += batch_to_shuffle

            if index > data_len:
                index = 0

        except KeyboardInterrupt:
            pass
        except ValueError as e:
            print(e)


def data_generator_labels(file_handle, data_file_keys, yield_labels=False, batch_size=1, nb_frames=16,
                          samples_to_cache=2000):
    index = 0
    data_len = len(file_handle[data_file_keys['classes']])
    print(f'Data len {data_len}')
    while True:
        try:
            print(f'Processing {index}:{index+samples_to_cache} part of the data')
            indices = np.arange(samples_to_cache)

            if not yield_labels:
                indices = indices[file_handle[data_file_keys['labels']][index:index + samples_to_cache] == 1]
                print(len(indices))

            classes = file_handle[data_file_keys['classes']][index:index + samples_to_cache][indices]
            videos = file_handle[data_file_keys['frames']][index:index + samples_to_cache][indices]

            if yield_labels:
                labels = file_handle[data_file_keys['labels']][index:index + samples_to_cache][indices]

            for i in range(0, samples_to_cache, batch_size):
                try:
                    batch_classes = classes[i:i + batch_size]

                    if len(batch_classes) == 0:
                        print("\n No more data in the batch \n\n")
                        break

                    batch_videos = videos[i:i + batch_size]
                    frames = create_frames_array(batch_videos, nb_frames=nb_frames)

                    if yield_labels:
                        batch_labels = labels[i:i + batch_size]
                        one_hot = labes_to_onehot(batch_labels)
                        yield frames, [batch_classes, one_hot]

                    yield frames, batch_classes

                except ValueError:
                    traceback.print_exc()

            index += samples_to_cache
            if index > data_len:
                index = 0
        except ValueError:
            traceback.print_exc()

