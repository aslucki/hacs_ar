import os
import tempfile

import numpy as np
import pandas as pd

import hacs.bin.process_data as processing
from hacs.processing import save_to_hdf5
from hacs.processing.frame_extractor import video_to_frames

VIDEOS = os.path.join(os.path.dirname(__file__), 'data', 'hacs_videos')
METADATA = os.path.join(os.path.dirname(__file__), 'data', 'hacs_test.csv')


def get_fasttext_mapping():
    class DummyDict(dict):
        def __getitem__(self, *args, **kwargs):
            return np.zeros(40)

    dummy = DummyDict()
    return dummy


def compare_known_data(processing_output, metadata, yt_id):
    known_record = metadata[metadata.youtube_id == yt_id]
    index = known_record.index.values[-1]
    video_path = os.path.join(os.path.dirname(__file__), 'data', 'hacs_videos',
                              known_record.classname.values[-1].replace(" ", "_"),
                              "v_{}_{}.mp4".format(index, yt_id))

    assert processing_output['labels'][index] == known_record.label.values[-1]
    assert yt_id in processing_output['video_ids'][index]


def test_generate_batch():

    metadata = pd.read_csv(METADATA)

    classes = sorted(np.unique(metadata[~pd.isnull(metadata.classname)].classname))
    classes_mapping = processing.create_mappings(classes)
    fasttext_mapping = get_fasttext_mapping()

    train_data = metadata[metadata.subset == 'training']

    train_gen = processing.generate_batches(VIDEOS,
                                            classes_mapping,
                                            fasttext_mapping,
                                            metadata=train_data)

    batch = next(train_gen)
    for k in ['class_names', 'labels', 'video_ids', 'one_hot', 'fasttext_vector', 'frames']:
        assert k in batch.keys()

    yt_id = 'a2X2hz1G6i8'
    compare_known_data(batch, train_data, yt_id)

    yt_id = '0O_qMHxBfXg'
    compare_known_data(batch, train_data, yt_id)

    with tempfile.TemporaryDirectory() as temp_dir:
        filename = "sample.h5"
        output_path = os.path.join(temp_dir, filename)

        save_to_hdf5(batch, output_file=output_path)
        assert os.path.isfile(output_path)