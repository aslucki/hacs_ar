import os
import tempfile

import h5py
import numpy as np

from hacs.processing import save_to_hdf5
from hacs.processing.frame_extractor import encode_frames


def test_save_to_hdf5():

    data = {
        "arrays": np.zeros((100, 20), dtype=np.float32),
        "strings": ["text"] * 100
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        filename = "sample.h5"
        output_path = os.path.join(temp_dir, filename)

        save_to_hdf5(data, output_file=output_path)
        assert os.path.isfile(output_path)

        save_to_hdf5(data, output_file=output_path)

        with h5py.File(output_path, 'r') as f:
            assert sorted(list(f.keys())) == sorted(data.keys())
            assert len(f['arrays']) == 200


def generate_frames():
    frames = np.ones(shape=(40, 200, 200, 3), dtype=np.uint8)

    return frames


def test_save_and_read_hdf5():

    frames = generate_frames()

    videos = []
    for _ in range(10):
        encoded = encode_frames(frames)
        videos.append(encoded)
    videos = np.asarray(videos)

    data = {"frames": videos,
            "names": np.asarray(['name']*len(videos))}

    with tempfile.TemporaryDirectory() as temp_dir:
        filename = "sample.h5"
        output_path = os.path.join(temp_dir, filename)

        save_to_hdf5(data, output_file=output_path)
        assert os.path.isfile(output_path)
        save_to_hdf5(data, output_file=output_path)

        with h5py.File(output_path, 'r') as f:
            assert sorted(list(f.keys())) == sorted(data.keys())
            assert f['frames'][-1][0].tostring() == videos[-1][0].tostring()
