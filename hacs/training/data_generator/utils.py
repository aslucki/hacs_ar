from itertools import groupby, count

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


def smooth_labels(labels, factor=0.2):
    labels = np.asarray(labels)
    if factor >= 1:
        raise ValueError("Factor must be a number between 0 and 1")
    labels[labels == 1] = 1-factor
    labels[labels == 0] = factor/(labels.shape[-1] - 1)

    return labels


def create_frames_array(videos, nb_frames, frame_shape):
    frames = []
    for video in videos:
        decoded = decode_frames(video)
        if decoded.shape[1] < frame_shape[0]:
            print(f'Error occurred, the video has shape: {decoded.shape}')
            decoded = np.zeros(shape=(decoded.shape[0],) + frame_shape)

        last_frame = find_last_frame(decoded)
        start_frame = 0
        if last_frame > nb_frames:
            start_frame = np.random.randint(0, last_frame - nb_frames, size=1)[0]

        cropped = crop_frames(decoded[start_frame:start_frame + nb_frames])
        floating = convert_to_floats(cropped)
        frames.append(floating)

    return np.asarray(frames)
