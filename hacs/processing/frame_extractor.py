from collections import namedtuple

import av
import numpy as np
from PIL import Image


class FrameExtractor:

    @staticmethod
    def video_to_frames(file_path, image_size=None):
        """
        Extract individual frames from a video.
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

        del container
        return video_data(frames=np.asarray(all_frames),
                          keyframes=np.asarray(keyframes))
