import h5py
import numpy as np


def retrieve_n_last_id(datafile_path, n, key='video_ids'):
    with h5py.File(datafile_path, 'r') as f:
        last = f[key][-n]
    return last


def find_index(metadata, datafile_path):
    last_item = retrieve_n_last_id(datafile_path=datafile_path, n=1)
    found_index = int(last_item.split('_')[0])

    return metadata.index.get_loc(found_index)


def _convert_list(data):
    if isinstance(data, list):
        data = np.asarray(data)

    return data


def _create_dataset(file_handle, ds_name, data):
    data = _convert_list(data)

    try:
        dataset = \
            file_handle.create_dataset(ds_name,
                                       maxshape=(None,) + data.shape[1:],
                                       data=data)
    except TypeError:
        if issubclass(data.dtype.type, str):
            dtype = h5py.string_dtype(encoding='utf-8')
        else:
            raise TypeError

        dataset = file_handle.create_dataset(ds_name, shape=data.shape,
                                             maxshape=(None,) + data.shape[1:],
                                             dtype=dtype)
        dataset[:] = data

    return dataset


def _resize_dataset(dataset, rows_to_add):
    current_size = dataset.shape
    dataset.resize((current_size[0] + rows_to_add,) + current_size[1:])


def save_to_hdf5(data: dict, output_file):
    with h5py.File(output_file, 'a') as file_handle:
        for ds_name, values in data.items():
            if ds_name not in file_handle.keys():
                if ds_name == 'frames':
                    # TODO: hack, should be handled properly
                    values = np.void(values)
                _create_dataset(file_handle, ds_name, data=values)

            else:
                dataset = file_handle[ds_name]
                values = _convert_list(values)
                start_index = dataset.shape[0]
                _resize_dataset(dataset, values.shape[0])
                dataset[start_index:] = values





