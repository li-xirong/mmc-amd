from .utils import *
from .dataset import MultiDataset, SingleDataset


class DataOrganizer:
    @staticmethod
    def get_fundus_data(collection, aug_params=None, transform=None, if_test=False, cls_num=4, if_eval=False):
        path_file = join_path(collection, "ImageSets", "cfp.txt")
        imgs_path_list = load_pathfile(collection, "cfp-clahe-448x448", path_file)
        if (not if_test) or if_eval:
            label_file = join_path(collection, "annotations", "cfp.txt")
            labels_list = load_labelfile(label_file)
        else:
            labels_list = None
        if if_eval:
            return imgs_path_list, labels_list
        return SingleDataset(imgs_path_list, labels_list, aug_params, transform, if_test, cls_num)

    @staticmethod
    def get_oct_data(collection, aug_params=None, transform=None, if_test=False, cls_num=4, if_eval=False):
        path_file = join_path(collection, "ImageSets", "oct.txt")
        imgs_path_list = load_pathfile(collection, "oct-median3x3-448x448", path_file)
        if not if_test or if_eval:
            label_file = join_path(collection, "annotations", "oct.txt")
            labels_list = load_labelfile(label_file)
        else:
            labels_list = None
        if if_eval:
            return imgs_path_list, labels_list
        return SingleDataset(imgs_path_list, labels_list, aug_params, transform, if_test, cls_num)

    @staticmethod
    def get_mm_data(
            collection, loosepair=False, aug_params=None, transform=None, if_test=False, cls_num=4, if_eval=False,
            if_syn=False, syn_collection=None):
        def pair_sampler(
                imgs_f_path_list, imgs_o_path_list, eyeids_f_list, eyeids_o_list,
                labels_f_list=None, labels_o_list=None,
                loosepair=False, if_test=False, if_syn=False):
            """pair fundus and oct"""
            if if_test:
                loosepair = False
                labels_f_list = [None] * len(imgs_f_path_list)
                labels_o_list = [None] * len(imgs_o_path_list)
            fundus_zip = list(zip(imgs_f_path_list, eyeids_f_list, labels_f_list))
            oct_zip = list(zip(imgs_o_path_list, eyeids_o_list, labels_o_list))
            pairs_path_list = []
            labels_list = []
            for path_o, eyeid_o, label_o in oct_zip:
                for path_f, eyeid_f, label_f in fundus_zip:
                    if if_syn:
                        if ("trs" not in path_o) and ("ori" not in path_o) and ("flr" not in path_o) and \
                                ("trs" not in path_f) and ("ori" not in path_f) and ("flr" not in path_f):
                            continue
                    if loosepair:
                        if label_o == label_f:
                            pairs_path_list.append((path_f, path_o))
                            labels_list.append(label_f)
                    else:
                        if eyeid_o == eyeid_f:
                            pairs_path_list.append((path_f, path_o))
                            labels_list.append(label_f)
            if if_test:
                return pairs_path_list, None
            else:
                return pairs_path_list, labels_list

        path_file_f = join_path(collection, "ImageSets", "cfp.txt")
        imgs_f_path_list = load_pathfile(collection, "cfp-clahe-448x448", path_file_f)
        path_file_o = join_path(collection, "ImageSets", "oct.txt")
        imgs_o_path_list = load_pathfile(collection, "oct-median3x3-448x448", path_file_o)

        if if_syn:
            path_file_f_syn = join_path(syn_collection, "ImageSets", "cfp.txt")
            imgs_f_path_list.extend(load_pathfile(syn_collection, "cfp-clahe-448x448", path_file_f_syn))
            path_file_o_syn = join_path(syn_collection, "ImageSets", "oct.txt")
            imgs_o_path_list.extend(load_pathfile(syn_collection, "oct-median3x3-448x448", path_file_o_syn))

        eyeids_f_list = get_eyeid_batch(imgs_f_path_list)
        eyeids_o_list = get_eyeid_batch(imgs_o_path_list)

        if (not if_test) or if_eval:
            label_file_f = join_path(collection, "annotations", "cfp.txt")
            labels_f_list = load_labelfile(label_file_f)
            label_file_o = join_path(collection, "annotations", "oct.txt")
            labels_o_list = load_labelfile(label_file_o)
            if if_syn:
                label_file_f_syn = join_path(syn_collection, "annotations", "cfp.txt")
                labels_f_list.extend(load_labelfile(label_file_f_syn))
                label_file_o_syn = join_path(syn_collection, "annotations", "oct.txt")
                labels_o_list.extend(load_labelfile(label_file_o_syn))
        else:
            labels_f_list = None
            labels_o_list = None

        pairs_path_list, labels_list = pair_sampler(
            imgs_f_path_list, imgs_o_path_list, eyeids_f_list, eyeids_o_list, labels_f_list, labels_o_list,
            loosepair, if_test, if_syn)

        if if_eval:
            return pairs_path_list, labels_list

        return MultiDataset(pairs_path_list, labels_list, aug_params, transform, if_test, cls_num)
