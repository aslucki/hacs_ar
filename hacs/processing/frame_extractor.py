from collections import namedtuple

import numpy as np
import cv2


def video_to_frames(file_path, image_size=None, constant_num_frames=None):
    cap = cv2.VideoCapture(file_path)

    frames_cnt = 0
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        frames_cnt += 1
        if constant_num_frames and frames_cnt > constant_num_frames:
            break

        if image_size:
            frame = cv2.resize(frame, image_size)
        frames.append(frame)

    frames = np.asarray(frames)
    if constant_num_frames and len(frames) < constant_num_frames:
        padded_frames = np.ones((constant_num_frames,) + frames.shape[1:],
                                dtype=frames.dtype)
        padded_frames[:len(frames)] = frames
        frames = padded_frames

    cap.release()

    return frames


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
