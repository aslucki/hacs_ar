import os

import fasttext
import fire
import numpy as np
import pandas as pd
import pickle
import yaml


def calculate_vector(words, model):
    vector = []
    for word in words:
        vector.append(model[word])
    return np.mean(vector, axis=0)


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config


def process_classes(config_path):
    config = load_config(config_path)

    print(config['paths']['fasttext_model'])
    model = fasttext.load_model(config['paths']['fasttext_model'])
    print("Fasttext model loaded")

    metadata = pd.read_csv(config['paths']['metadata_file'])
    class_names = np.unique(metadata[~pd.isnull(metadata.classname)].classname)

    mapping = dict()
    for classname in class_names:
        mapping[classname] = calculate_vector(classname.lower().split(),
                                              model)

    save_path = config['paths']['fasttext_mapping']
    print("Processed data.\n Saving to {}".format(save_path))

    with open(save_path, 'wb') as f:
        pickle.dump(mapping, f)


if __name__ == "__main__":
    fire.Fire(process_classes)
