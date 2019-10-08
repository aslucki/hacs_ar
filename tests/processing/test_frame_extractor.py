import os

import numpy as np

from hacs.processing.frame_extractor import video_to_frames, encode_frames

VIDEO_PATH = os.path.join(os.path.dirname(__file__), 'data', 'video.mp4')


def test_extract_frames():

    for i in range(1, 3):
        size = 224*i*2

        try:
            output = video_to_frames(VIDEO_PATH, (size, size))
            assert output.frames.shape[1:] == (size, size, 3)
            assert len(output.frames) > len(output.keyframes)
            assert len(output.frames.shape) == 4

        except ValueError:
            pass


def test_multiple_query():

    output1 = video_to_frames(VIDEO_PATH, (100, 100))
    output2 = video_to_frames(VIDEO_PATH, (100, 100))

    assert np.all(output1.frames == output2.frames)


def test_encoding():
    frames = video_to_frames(VIDEO_PATH, (100, 100))
    encoded = encode_frames(frames.frames)

    assert b'\xff\xd8\xff\xe0' not in encoded
