from collections import namedtuple

import av
import numpy as np
import cv2
from PIL import Image


def video_to_frames(file_path, image_size=None, constant_num_frames=None):
    """
    Extract individual frames from a video.
    :param constant_num_frames:
    :param file_path: Path to a video file.
    :param image_size: Desired image dimensions.
        If not specified extracted frames will have the same dimensions as the original video.
    :return: Named tuple containing keyframes and all video frames.
    """
    if 0 in image_size:
        raise ValueError("Image size must be greater than 0")

    container = av.open(file_path)

    try:
        stream = container.streams.video[0]
    except IndexError:
        return None

    video_data = namedtuple('video_data', 'frames keyframes')
    all_frames = []
    keyframes = []
    for frame in container.decode(stream):

        img = frame.to_image()

        if image_size:
            img = img.resize(image_size, Image.ANTIALIAS)

        if frame.key_frame:
            keyframes.append(np.array(img, dtype=np.float32))

        all_frames.append(np.array(img, dtype=np.uint8))

    if stream:
        stream.close()
        del stream

    del container

    all_frames = np.asarray(all_frames)
    if constant_num_frames:
        if len(all_frames) >= constant_num_frames:
            all_frames = all_frames[:constant_num_frames]
        else:
            frames = np.ones((constant_num_frames,) + all_frames.shape[1:],
                             dtype=all_frames.dtype)
            frames[:len(all_frames)] = all_frames[:]
            all_frames = frames

    return video_data(frames=all_frames,
                      keyframes=np.asarray(keyframes))


def encode_frames(frames):
    output = []
    for frame in frames:
        img_str = cv2.imencode('.jpg', frame)[1].tostring()
        output.append(img_str)

    return np.asarray(output)


def decode_frames(frames):
    output = []
    for frame in frames:
        parsed = np.frombuffer(frame, np.uint8)
        image = cv2.imdecode(parsed, cv2.IMREAD_COLOR)
        output.append(image)

    return np.asarray(output)
