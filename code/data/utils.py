import os


def join_path(collection_path, sub_path, file):
    return os.path.join(collection_path, sub_path, file)


def load_labelfile(filepath):
    with open(filepath, "r")as fp:
        lines = fp.readlines()
    labels_list = [int(line.strip().split(" ")[-1]) for line in lines]
    return labels_list


def load_pathfile(collection, moda, filepath):
    with open(filepath, "r")as fp:
        lines = fp.readlines()
    imgs_path_list = [os.path.join(collection, 'ImageData', moda, line.strip() + '.jpg') for line in lines]
    return imgs_path_list


def get_eyeid(name):
    """
    + get eye-id from a single filename
    + This function is highly dependent on how the datafiles are named
    + '-trs' '-ori' '-flr' are the suffixes of synthetic images
    """
    return name[2:-4].split("_")[0].replace("-trs", "").replace("-ori", "").replace("-flr", "")


def get_eyeid_batch(imgs_path_list):
    """get eye-ids"""
    eyeids_list = []
    for item in imgs_path_list:
        eyeids_list.append(get_eyeid(os.path.split(item)[-1]))
    return eyeids_list
