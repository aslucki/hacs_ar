import os
import tempfile

import h5py
import numpy as np

from hacs.processing import save_to_hdf5


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


