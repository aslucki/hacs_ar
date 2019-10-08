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
        return min(indexes)

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
    start_y = np.random.randint(0, frames_size[0] - size[0], size=1)[0]
    start_x = np.random.randint(0, frames_size[1] - size[1], size=1)[0]

    cropped = []
    for frame in frames:
        cropped.append(frame[start_y:start_y + size[0],
                       start_x:start_x + size[1], :])

    return np.asarray(cropped)


def labes_to_onehot(labels):
    one_hot = np.zeros(shape=(len(labels), 2))
    one_hot[labels == 1, 1] = 1
    one_hot[labels == -1, 0] = 1

    return one_hot


def data_generator(file_handle, data_file_keys, yield_labels=False, batch_size=1, nb_frames=16):
    index = 0
    while True:
        try:
            classes = file_handle[data_file_keys['classes']][index:index + batch_size]
            frames = []
            for i in range(batch_size):
                all_frames = file_handle[data_file_keys['frames']][index + i]
                decoded = decode_frames(all_frames)
                last_frame = find_last_frame(decoded)
                start_frame = np.random.randint(0, last_frame - nb_frames, size=1)[0]
                cropped = crop_frames(decoded[start_frame:start_frame + nb_frames])
                frames.append(cropped)

            if yield_labels:
                labels = file_handle[data_file_keys['labels']][index:index + batch_size]
                one_hot = labes_to_onehot(labels)
                index += batch_size
                yield np.asarray(frames), [classes, one_hot]

            index += batch_size
            yield np.asarray(frames), classes

        except ValueError:
            index = 0


def create_frames_array(videos, nb_frames):

    frames = []
    for video in videos:
        decoded = decode_frames(video)
        last_frame = find_last_frame(decoded)
        start_frame = np.random.randint(0, last_frame - nb_frames, size=1)[0]
        cropped = crop_frames(decoded[start_frame:start_frame + nb_frames])
        frames.append(cropped)

    return np.asarray(frames)


def data_generator_with_shuffle(file_handle, data_file_keys, yield_labels=False, batch_size=1, nb_frames=16,
                                batch_to_shuffle=1000):
    index = 0
    while True:
        try:
            indices = np.arange(batch_to_shuffle, dtype=int)
            np.random.shuffle(indices)

            classes = file_handle[data_file_keys['classes']][index:index + batch_to_shuffle][indices]
            videos = file_handle['frames'][index:index + batch_to_shuffle][indices]

            if yield_labels:
                labels = file_handle[data_file_keys['labels']][index:index + batch_to_shuffle][indices]

            for i in range(0, batch_to_shuffle, batch_size):
                batch_classes = classes[i:i + batch_size]
                batch_videos = videos[i:i + batch_size]
                frames = create_frames_array(batch_videos, nb_frames=nb_frames)

                if yield_labels:
                    batch_labels = labels[i:i + batch_size]
                    one_hot = labes_to_onehot(batch_labels)
                    yield frames, [classes, one_hot]

                yield frames, batch_classes

            index += batch_to_shuffle
        except ValueError:
            index = 0
