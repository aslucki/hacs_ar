import fire
import os
import pickle
import shutil
import sys
import yaml

import numpy as np
import tensorflow as tf

from hacs.training.trainers import C3DTrainer
from hacs.models.c3d import get_model


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config


def copy_config(config_path, experiment_dir):
    shutil.copy(config_path, experiment_dir)


def make_experiment_dir(output_dir, model_name):
    path = os.path.join(output_dir, model_name)
    os.makedirs(path,  exist_ok=True)

    return path


def save_history(experiment_dir, history):
    history_files = [file for file in os.listdir(experiment_dir)
                     if 'history' in file]

    suffix = len(history_files)
    with open(os.path.join(experiment_dir, f"history_{suffix}.pkl"), 'wb') as f:
        pickle.dump(history.history, f)


def train_model(config_path):
    config = load_config(config_path)
    model = get_model(config['training']['model_name'],
                      config['training']['use_negative_labels'])

    experiment_output_dir = make_experiment_dir(config['training']['output_dir'],
                                                config['training']['model_name'])
    copy_config(config_path, experiment_output_dir)
    trainer = C3DTrainer(model=model,
                         use_labels=config['training']['use_negative_labels'],
                         data_file_keys=config['training']['data_file_keys'],
                         output_dir=experiment_output_dir,
                         learning_rate=config['training']['learning_rate'])

    trainer.compile_model(optimizer=config['training']['optimizer'],
                          losses=config['training']['losses'],
                          metrics=config['training']['metrics'])

    training_file = config['training']['training_file']
    validation_file = config['training']['validation_file']

    try:
        history = trainer.train(training_file, validation_file, epochs=config['training']['epochs'],
                                batch_size=config['training']['batch_size'])

    except KeyboardInterrupt:
        sys.exit(1)

    finally:
        save_history(experiment_output_dir, history)


if __name__ == '__main__':
    np.random.seed(42)
    tf.random.set_seed(42)

    fire.Fire(train_model)
