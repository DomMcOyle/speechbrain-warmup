"""
Downloads and creates data manifest files for AudioMNIST (digit classification).
By default, The following script splits the dataset with the official split of 90-10
where the test set is made from the audios with indexes 0 to 4. The functions
also allow to extract a validation set with a desired percentage over the full dataset.
Authors:
 * Domenico Dell'Olio, 2023
"""

import os
import json
import shutil
import random
import logging
import re
from speechbrain.utils.data_utils import get_all_files, download_file
from speechbrain.dataio.dataio import read_audio

logger = logging.getLogger(__name__)
AUDIO_MNIST_URL = 'https://github.com/Jakobovski/free-spoken-digit-dataset/archive/refs/heads/master.zip'
SAMPLERATE = 8000


def prepare_audio_mnist(
    data_folder,
    save_json_train,
    save_json_valid,
    save_json_test,
    split_ratios=[0.8,0.1,0.1],
    do_random_split=False
):
    """
    Prepares the json files for the AudioMNIST.

    Downloads the dataset if it is not found in the `data_folder`.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the AudioMNIST dataset is stored.
    save_json_train : str
        Path where the train data specification file will be saved.
    save_json_valid : str
        Path where the validation data specification file will be saved.
    save_json_test : str
        Path where the test data specification file will be saved.
    split_ratios: list
        list of values in [0,1] summing up to 1. It indicates the percentage of the
        dataset in train, validation and test respectively. By default, it follows the
        official split of 90-10 where the test set is made of the examples with indexes 0-4
        and extracts a further 10% from the training set as validation set.
    do_random_split: Bool
        Boolean indicating whether to perform a random split on the dataset or keep it deterministic.
        With the latter choice the test set is chosen by taking the respective percentage of the
        data samples with the lowest index. Then, proceding from the lowest remaining index
        ,the validation set is extracted. Otherwise, samples are extracted randomly within the same speaker
        and digit.

    Example
    -------
    #>>> data_folder = '/path/to/audio_mnist'
    #>>> prepare_audio_mnist(data_folder, 'train.json', 'valid.json', 'test.json')
    """

    # Check if this phase is already done (if so, skip it)
    if skip(save_json_train, save_json_valid, save_json_test):
        logger.info("Preparation completed in previous run, skipping.")
        return

    # If the dataset doesn't exist yet, download it
    train_folder = os.path.join(data_folder, "free-spoken-digit-dataset-master", "recordings")
    if not check_folders(train_folder):
        download_audio_mnist(data_folder)

    # List files and create manifest from list
    logger.info(
        f"Creating {save_json_train}, {save_json_valid}, and {save_json_test}"
    )
    extension = [".wav"]
    wav_list = get_all_files(train_folder, match_and=extension)
    # Split the signal list into train, valid, and test sets.
    data_split = split_sets(wav_list, split_ratios, do_random_split)
    # Creating json files
    create_json(data_split["train"], save_json_train)
    create_json(data_split["valid"], save_json_valid)
    create_json(data_split["test"], save_json_test)


def create_json(wav_list, json_file):
    """
    Creates the json file given a list of wav files.

    Arguments
    ---------
    wav_list : list of str
        The list of wav files.
    json_file : str
        The path of the output json file
    """
    # Processing all the wav files in the list
    json_dict = {}
    for wav_file in wav_list:

        # Reading the signal (to retrieve duration in seconds)
        signal = read_audio(wav_file)
        duration = signal.shape[0] / SAMPLERATE

        # Manipulate path to get relative path and example_id
        path_parts = wav_file.split(os.path.sep)
        example_id, _ = os.path.splitext(path_parts[-1])
        relative_path = os.path.join("{data_root}", *path_parts[-3:])

        # Getting digit label from example_id
        label = example_id.split("_")[0]
        # getting speaker information from example_id
        speaker = example_id.split('_')[1]

        # Create entry for this utterance
        json_dict[example_id] = {
            "wav": relative_path,
            "length": duration,
            'speaker': speaker,
            "label": label,
        }

    # Writing the dictionary to the json file
    with open(json_file, mode="w") as json_f:
        json.dump(json_dict, json_f, indent=2)

    logger.info(f"{json_file} successfully created!")


def skip(*filenames):
    """
    Detects if the data preparation has been already done.
    If the preparation has been done, we can skip it.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """
    for filename in filenames:
        if not os.path.isfile(filename):
            return False
    return True

def check_folders(*folders):
    """Returns False if any passed folder does not exist."""
    for folder in folders:
        if not os.path.exists(folder):
            return False
    return True


def split_sets(wav_list, split_ratios, do_random_split):
    """ splits the wav files list into training, validation and test lists.
    It allows both to create a deterministic split in order to emulate the
    official train-test split (examples with indexes in 0-4 are used for test)
    as well as a random split which selects random examples within the same speaker
    and digit. In this way, the split is always stratified w.r.t. speakers and
    digits.

    Arguments
    ---------
    wav_lst : list
        list of all the signals in the dataset
    split_ratios: list
        List composed of three floats that sets split ratios for train, valid,
        and test sets, respectively. For instance split_ratio=[.8, .1, .1] will
        assign 80% of the sentences to training, 10% for validation, and 10%
        for test.
    do_random_split: Bool
        Boolean indicating whether to select the examples randomly within the same speaker
        and digit or if to start separating the dataset starting from the lowest indexed examples
        first to obtain the test set, and then for validation.

    Returns
    ------
    dictionary containing train, valid, and test splits.
    """
    # if we want the random split we sort the samples for each speaker randomly,
    # else we sort them by index value
    sort_func = (lambda y: random.sample(y, len(y))) if do_random_split else \
        (lambda x: sorted(x, key=(lambda y: int(y.split('_')[-1].split('.')[0]))))
    # get the list of speakers
    speakers = set([f.split('\\')[-1].split('_')[1] for f in wav_list])

    # create a nested dictionary where for each speaker we group the wav file paths
    # by the digit they represent
    tree_info = {s:
                     {d: sort_func([w for w in wav_list if bool(re.match(f'^{d}_{s}', w.split('\\')[-1]))])
                      for d in range(0, 10)}
                 for s in speakers}

    # extract the desired number of test examples for each speaker and digit
    test_indexes = {s: {d: int(split_ratios[2]*len(tree_info[s][d])) for d in range(0, 10)} for s in speakers}

    # extract the desired number of validation examples for each speaker and digit
    valid_indexes = {s: {d: int(split_ratios[1] * len(tree_info[s][d])) for d in range(0, 10)} for s in speakers}
    # iterate on each speaker and digit
    data_split = {'train': [], 'valid': [], 'test': []}
    for s in tree_info.keys():
        for d in range(0, 10):
            # add the examples to the test set, and remove them from the current list
            data_split['test'].extend(tree_info[s][d][:test_indexes[s][d]])
            tree_info[s][d] = tree_info[s][d][test_indexes[s][d]:]
            # add the examples to the validation set
            data_split['valid'].extend(tree_info[s][d][:valid_indexes[s][d]])
            # add the rest of the examples to the training set
            data_split['train'].extend(tree_info[s][d][valid_indexes[s][d]:])
    return data_split


def download_audio_mnist(destination):
    """Download dataset and unpack it.

    Arguments
    ---------
    destination : str
        Place to put dataset.
    """
    train_archive = os.path.join(destination, "audioMNIST.zip")
    download_file(AUDIO_MNIST_URL, train_archive, unpack=True)
    shutil.unpack_archive(train_archive, destination)


prepare_audio_mnist('..\\..\\..\\',
                    "..\\..\\..\\free-spoken-digit-dataset-master\\splits\\train.json",
                    "..\\..\\..\\free-spoken-digit-dataset-master\\splits\\val.json",
                    "..\\..\\..\\free-spoken-digit-dataset-master\\splits\\test.json")