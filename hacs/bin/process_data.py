import os
import threading

import fire
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import yaml

from hacs.processing import FrameExtractor, save_to_hdf5
from hacs.processing.h5py_io import find_index


def get_start_index(output_path, metadata):
    start_index = 0
    if os.path.isfile(output_path):
        start_index = find_index(metadata, output_path)

    return start_index


def load_fasttext_mapping(path):
    with open(path, 'rb') as f:
        mapping = pickle.load(f)

    return mapping


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config


def _construct_video_path(input_dir, youtube_id, classname, index):

    path = os.path.join(input_dir, classname.replace(" ", "_"),
                        "v_{}_{}.mp4".format(index, youtube_id))

    if os.path.isfile(path):
        return path

    return None


def create_mappings(classes):
    class_to_index = {name: i for i, name in enumerate(classes)}
    index_to_class = {i: name for i, name in enumerate(classes)}

    return {
        'num_classes': len(classes),
        'class_to_index': class_to_index,
        'index_to_class': index_to_class
    }


def index_to_onehot(num_classes, index):
    one_hot = np.zeros(num_classes)
    one_hot[index] = 1

    return one_hot


def classes_to_one_hot(mapping, class_names):
    one_hot_labels = []
    for classname in class_names:
        index = mapping['class_to_index'][classname]
        one_hot_labels.append(index_to_onehot(mapping['num_classes'], index))

    return np.asarray(one_hot_labels)


def classes_to_fasstext(fasttext_mapping, class_names):
    vectors = []
    for classname in class_names:
        vectors.append(fasttext_mapping[classname])

    return np.asarray(vectors)


def _insert_video_output(index, video_path, img_size, output):
    result = FrameExtractor.video_to_frames(video_path,
                                            image_size=img_size)
    if result is not None:
        output[index] = result.frames

    return None


def process_videos(paths, threads_num=8, img_size=(200, 200)):

    processed_videos = [None] * len(paths)
    threads = []
    for i, path in enumerate(paths):
        if i != 0 and i % threads_num == 0:
            for thread in threads:
                thread.join()
        thread = threading.Thread(target=_insert_video_output,
                                  args=(i, str(path), img_size, processed_videos))
        threads.append(thread)
        thread.start()

    [t.join() for t in threads]

    return np.asarray(processed_videos)


def generate_batches(base_path, class_mapping, fasttext_mapping, metadata, batch_size=32,
                     threads_num=8, img_size=(200, 200), start_index=0):

    print("Starting processing from {}th row in metadata".format(start_index))
    for i in tqdm(range(start_index, len(metadata), batch_size)):
        data = metadata[i:i + batch_size]

        video_paths = []
        class_names = []
        labels = []
        video_ids = []

        for youtube_id, classname, index, label \
                in zip(data.youtube_id, data.classname, data.index, data.label):
            video_path = _construct_video_path(base_path, youtube_id, classname, index)
            video_paths.append(video_path)

            class_names.append(classname)
            labels.append(label)
            video_ids.append(youtube_id)

        video_paths = np.asarray(video_paths)
        valid_indices = np.where(video_paths != None)
        valid_paths = video_paths[valid_indices]

        processed_videos = None
        processed_videos = process_videos(valid_paths, threads_num=threads_num, img_size=img_size)
        new_valid_indices = np.asarray([i for i, video in enumerate(processed_videos) if video is not None])
        class_names = np.asarray(class_names)[valid_indices][new_valid_indices]

        batch = {
            'class_names': class_names,
            'labels': np.asarray(labels)[valid_indices][new_valid_indices],
            'video_ids': np.asarray(video_ids)[valid_indices][new_valid_indices],
            'one_hot': classes_to_one_hot(class_mapping, class_names),
            'fasttext_vector': classes_to_fasstext(fasttext_mapping, class_names),
            'frames': processed_videos[new_valid_indices]
        }

        yield batch


def save_batches(generator, output_file):
    while True:
        try:
            batch = next(generator)
            save_to_hdf5(batch, output_file)
        except StopIteration:
            break


def process_data(config_path):
    config = load_config(config_path)

    metadata = pd.read_csv(config['paths']['metadata_file'])
    classes = sorted(np.unique(metadata[~pd.isnull(metadata.classname)].classname))
    class_mapping = create_mappings(classes)
    fasttext_mapping = load_fasttext_mapping(config['paths']['fasttext_mapping'])

    trainig_data = metadata[metadata.subset == 'training']
    validation_data = metadata[metadata.subset == 'validation']

    train_start_index = get_start_index(config['paths']['training_file'], trainig_data)
    validation_start_index = get_start_index(config['paths']['validation_file'], validation_data)

    trainig_gen = generate_batches(config['paths']['input_dir'],
                                   class_mapping,
                                   fasttext_mapping,
                                   metadata=trainig_data,
                                   batch_size=config['processing']['batch_size'],
                                   img_size=config['processing']['img_size'],
                                   threads_num=config['processing']['threads_num'],
                                   start_index=train_start_index)

    validation_gen = generate_batches(config['paths']['input_dir'],
                                      class_mapping,
                                      fasttext_mapping,
                                      metadata=validation_data,
                                      batch_size=config['processing']['batch_size'],
                                      img_size=config['processing']['img_size'],
                                      threads_num=config['processing']['threads_num'],
                                      start_index=validation_start_index)

    save_batches(trainig_gen, config['paths']['training_file'])
    save_batches(validation_gen, config['paths']['validation_file'])


if __name__ == "__main__":
    fire.Fire(process_data)
