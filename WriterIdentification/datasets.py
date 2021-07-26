import torch
import os
import cv2
import numpy as np
import random
import torch.utils.data as D
import json

from glob import glob
from iam_gt_pairs import *
from handwriting_functions import train_preprocess, test_preprocess


modes = ["train", "val", "test"]
balanced_classes: [1, 3, 5, 6, 7, 10, 11, 14, 15, 20, 21, 24, 25, 26, 27, 28, 29, 31, 32, 38, 43, 49, 50, 53, 57, 58, 61, 71, 73, 74, 85, 87, 88, 89, 93, 94, 95, 97, 100, 107, 129, 146, 156, 162, 168, 177, 178, 179, 185, 187, 284, 285, 296, 297, 301, 309, 320, 325, 326, 332, 343, 350, 351, 352, 354, 359, 366, 374, 379, 391, 399, 400, 402, 404, 415, 417, 424, 428, 439, 449, 454, 457, 461, 463, 474, 489, 490, 496]
IAM_NUM_CLASSES = 500
NUM_CLASSES = 2
IAM_CLASS = 0
TARGET_CLASS = 1


class IAM_Words(D.Dataset):
    """
    src_path:       Images base path
    gt_path:        Path to the ground truth text file that contains the 500 writer IDs, the image names and the text content
    mode:           One of ['trian', 'val', 'test']
    width, height:  Images shape
    chosen_labels:  Which writer labels to take into consideration
    mode_start_ratio and mode_end_ratio: To determine where the mode begins and ends
    (e.g: 0.8. 1.0 --> Only the last 20% of the dataset paths are taken into consideration in this mode)
    """

    def __init__(self, src_path, gt_path, mode, mode_start_ratio, mode_end_ratio, width, height, chosen_labels):
        super(IAM_Words, self).__init__()
        global modes, balanced_classes, IAM_NUM_CLASSES
        gt_lines = None
        with open(gt_path, 'r') as f:
            gt_lines = f.readlines()

        assert len(gt_lines) > 0, "Invalid gt file"
        assert mode_start_ratio < mode_end_ratio and mode_start_ratio >= 0, "Invalid mode ratios"
        assert len(chosen_labels) > 0, "Invalid list of class labels"
        assert mode in modes, "Invalid mode"

        typ = glob(os.path.join(src_path, '*'))[0].split('.')[-1]        
        chosen_ids = [label2wid[lbl] for lbl in chosen_labels]
        self.paths = [os.path.join(src_path, f"{l.split(',')[-1].split(' ')[0]}.{typ}") for l in gt_lines if int(l.split(',')[0]) in chosen_ids]
        self.labels = [torch.tensor(wid2label[int(l.split(',')[0])]).long() for l in gt_lines if int(l.split(',')[0]) in chosen_ids]
        if len(chosen_labels) == len(balanced_classes):
            self.labels = [torch.tensor(labels500_2labels88[l.item()]).long() for l in self.labels]
        self.paths = self.paths[int(mode_start_ratio*len(self.paths)) : int(mode_end_ratio*len(self.paths))]
        self.labels = self.labels[int(mode_start_ratio*len(self.labels)) : int(mode_end_ratio*len(self.labels))]
        
        self.width = width
        self.height = height
        self.mode = mode


    def __getitem__(self, idx):
        x = cv2.imread(self.paths[idx], 0)
        if self.mode is modes[0]:
            x = train_preprocess(x, self.width, self.height)
        else:
            x = test_preprocess(x, self.width, self.height)
        x = torch.from_numpy(x).unsqueeze(0).float()
        y = self.labels[idx]
        return x, y


    def __len__(self):
        return len(self.paths)



class Two_Classes_Words(D.Dataset):
    """
    iam_words_dir:      IAM base path
    gt_path:            Path to the ground truth text file that contains the writer IDs, the image names and the text content
    target_words_dir:   Directory of the target writer's word images
    mode:               One of ['trian', 'val', 'test']
    width, height:      Images shape
    chosen_labels:      Which writer labels to take into consideration
    mode_start_ratio and mode_end_ratio: To determine where the mode begins and ends
    (e.g: 0.8. 1.0 --> Only the last 20% of the dataset paths are taken into consideration in this mode)
    """

    def __init__(self, iam_words_dir, gt_path, json_path, target_words_dir,
                 mode, mode_start_ratio, mode_end_ratio, width, height):

        super(Two_Classes_Words, self).__init__()
        global modes

        assert bool(json_path) != bool(gt_path), "json_path and gt_path are both existing. Please use only one of them"
        assert mode_start_ratio < mode_end_ratio and mode_start_ratio >= 0, "Invalid mode ratios"
        assert mode in modes, "Invalid mode"

        self.chosen_ids = None
        if json_path is None and gt_path is not None:
            self.init_from_gt(gt_path, target_words_dir, iam_words_dir)
        elif json_path is not None and gt_path is None:
            self.init_from_json(json_path, mode, mode_start_ratio, mode_end_ratio, target_words_dir)

        paths_labels = list(zip(self.paths, self.labels))
        random.shuffle(paths_labels)
        self.paths, self.labels = zip(*paths_labels)
        self.paths = self.paths[int(len(self.paths)*mode_start_ratio):int(len(self.paths)*mode_end_ratio)]
        self.labels = self.labels[int(len(self.labels)*mode_start_ratio):int(len(self.labels)*mode_end_ratio)]

        self.width = width
        self.height = height
        self.mode = mode


    def init_from_json(self, json_path, mode, mode_start_ratio, mode_end_ratio, target_words_dir):
        global IAM_CLASS, TARGET_CLASS
        assert json_path != None, "Invalid json path"

        # Add "IAM class" samples
        with open(json_path, 'r') as f:
            data = json.load(f)
            self.chosen_ids = list(data.keys()) 
            target_paths = glob(os.path.join(target_words_dir, '*'))
            num_samples_per_writer = 1+len(target_paths)//len(self.chosen_ids)

            self.paths, self.labels = list(), list()
            for label, wid in enumerate(list(data.keys())):
                for i, p in enumerate(data[wid]):
                    if i >= num_samples_per_writer:
                        break
                    self.paths.append(p)
                    self.labels.append(torch.tensor(IAM_CLASS).long())
        
        # Add target class samples
        self.paths += target_paths
        self.labels += [torch.tensor(TARGET_CLASS).long()] * len(target_paths)


    def init_from_gt(self, gt_path, target_words_dir, iam_words_dir):
        global IAM_CLASS, TARGET_CLASS

        self.typ1 = glob(os.path.join(iam_words_dir, '*'))[0].split('.')[-1]
        self.typ2 = glob(os.path.join(target_words_dir, '*'))[0].split('.')[-1]

        if str(len(balanced_classes)) in gt_path:
            self.chosen_ids = [label2wid[labels88_2labels500[lbl]] for lbl in balanced_classes]
        elif len(balanced_classes) == IAM_NUM_CLASSES:
            self.chosen_ids = [label2wid[lbl] for lbl in balanced_classes]
        assert self.chosen_ids is not None, "Invalid gt_path"

        self.paths, self.labels = list(), list()
        target_paths = glob(os.path.join(target_words_dir, '*'))
        num_samples_per_writer = len(target_paths)//len(self.chosen_ids)

        with open(gt_path, 'r') as f:
            gt_lines = f.readlines()
            assert len(gt_lines) > 0, "Invalid gt file"

            # Add "IAM class" samples
            for wid in self.chosen_ids:
                for i, line in enumerate(gt_lines):
                    if i >= num_samples_per_writer:
                        break
                    if int(line.split(',')[0]) == wid:
                        self.paths.append(os.path.join(iam_words_dir, f"{line.split(',')[-1].split(' ')[0]}.{self.typ1}"))
                        self.labels.append(torch.tensor(IAM_CLASS).long())
            
            # Add target class samples
            self.paths += target_paths
            self.labels += [torch.tensor(TARGET_CLASS).long()] * len(target_paths)


    def __getitem__(self, idx):
        x = cv2.imread(self.paths[idx], 0)
        if self.mode is modes[0]:
            x = train_preprocess(x, self.width, self.height)
        else:
            x = test_preprocess(x, self.width, self.height)
        x = torch.from_numpy(x).unsqueeze(0).float()
        y = self.labels[idx]
        return x, y


    def __len__(self):
        return len(self.paths)

    

class One_VS_Rest_IAM(D.Dataset):
    """
    json_path:          Path to the ground truth json file that contains the writer IDs their word images paths
    target_wid:         Target writer ID 
    width, height:      Images shape
    mode:               One of ['train', 'val', 'test']
    mode_start_ratio and mode_end_ratio: To determine where the mode begins and ends
    (e.g: 0.8. 1.0 --> Only the last 20% of the dataset paths are taken into consideration in this mode)
    """

    def __init__(self, target_wid, json_path, mode, mode_start_ratio,
                 mode_end_ratio, width, height):

        super(One_VS_Rest_IAM, self).__init__()
        global modes

        assert json_path is not None, "'json_path' cannot be none"
        assert mode_start_ratio < mode_end_ratio and mode_start_ratio >= 0, "Invalid mode ratios"
        assert mode in modes, "Invalid mode"

        self.chosen_ids = None
        self.init_from_json(json_path, mode, mode_start_ratio, mode_end_ratio, target_wid)

        paths_labels = list(zip(self.paths, self.labels))
        random.shuffle(paths_labels)
        self.paths, self.labels = zip(*paths_labels)
        self.paths = self.paths[int(len(self.paths)*mode_start_ratio):int(len(self.paths)*mode_end_ratio)]
        self.labels = self.labels[int(len(self.labels)*mode_start_ratio):int(len(self.labels)*mode_end_ratio)]

        self.width = width
        self.height = height
        self.mode = mode


    @staticmethod
    def find_wid_with_maximum_paths(data):
        wids = list(data.keys())
        max_wid = wids[0]
        max_len_paths = data[max_wid]

        for wid, paths in data.items():
            if len(paths) > len(max_len_paths):
                max_wid = wid
                max_len_paths = paths

        print(f"Target WID is {max_wid} with {len(max_len_paths)} samples")
        return max_wid


    def init_from_json(self, json_path, mode, mode_start_ratio, mode_end_ratio, target_wid):
        global IAM_CLASS, TARGET_CLASS

        self.paths, self.labels = list(), list()
        
        with open(json_path, 'r') as f:
            data = json.load(f)
            if target_wid == -1:
                target_wid = One_VS_Rest_IAM.find_wid_with_maximum_paths(data)
            self.chosen_ids = list(data.keys())
            
            # Add target class samples
            target_paths = data[target_wid]
            self.paths += target_paths
            self.labels += [torch.tensor(TARGET_CLASS).long()] * len(target_paths)
            num_samples_per_writer = 1+len(target_paths)//(len(self.chosen_ids)-1)

            # Add "rest of IAM class" samples
            for label, wid in enumerate(list(data.keys())):
                if wid == target_wid:
                    continue
                for i, p in enumerate(data[wid]):
                    if i >= num_samples_per_writer:
                        break
                    self.paths.append(p)
                    self.labels.append(torch.tensor(IAM_CLASS).long())


    def __getitem__(self, idx):
        x = cv2.imread(self.paths[idx], 0)
        if self.mode is modes[0]:
            x = train_preprocess(x, self.width, self.height)
        else:
            x = test_preprocess(x, self.width, self.height)
        x = torch.from_numpy(x).unsqueeze(0).float()
        y = self.labels[idx]
        return x, y


    def __len__(self):
        return len(self.paths)



class SiameseDataset(D.Dataset):
    """
    iam_words_dir:      IAM base path
    gt_path:            Path to the ground truth text file that contains the writer IDs, the image names and the text content
    target_words_dir:   Directory of the target writer's word images
    mode:               One of ['trian', 'val', 'test']
    width, height:      Images shape
    chosen_labels:      Which writer labels to take into consideration
    mode_start_ratio and mode_end_ratio: To determine where the mode begins and ends
    (e.g: 0.8. 1.0 --> Only the last 20% of the dataset paths are taken into consideration in this mode)
    """

    iam_class = 0
    target_class = 1

    def __init__(self, iam_words_dir, gt_path, json_path, target_words_dir,
                 mode, mode_start_ratio, mode_end_ratio, width, height):
        super(SiameseDataset, self).__init__()
        global modes, balanced_classes, IAM_NUM_CLASSES

        assert (json_path is None and gt_path is not None) or (json_path is not None and gt_path is None), "json_path and gt_path are both existing. Please use only one of them"
        assert mode_start_ratio < mode_end_ratio and mode_start_ratio >= 0, "Invalid mode ratios"
        assert mode in modes, "Invalid mode"

        self.chosen_ids = None
        self.target_paths, self.target_labels = None, None
        self.general_paths, self.general_labels = list(), list()
        if json_path is None and gt_path is not None:
            self.init_from_gt(gt_path, target_words_dir, iam_words_dir)
        elif json_path is not None and gt_path is None:
            self.init_from_json(json_path, mode, mode_start_ratio, mode_end_ratio,
                target_words_dir)

        self.target_paths, self.target_labels = self.shuffle_pairs(self.target_paths,
            self.target_labels, mode_start_ratio, mode_end_ratio)
        self.general_paths, self.general_labels = self.shuffle_pairs(self.general_paths,
            self.general_labels, mode_start_ratio, mode_end_ratio)

        self.width = width
        self.height = height
        self.mode = mode


    def init_from_json(self, json_path, mode, mode_start_ratio, mode_end_ratio, target_words_dir):
        assert json_path != None, "Invalid json path"

        # Add "IAM class" samples
        with open(json_path, 'r') as f:
            data = json.load(f)
            self.chosen_ids = list(data.keys()) 
            self.target_paths = glob(os.path.join(target_words_dir, '*'))
            self.target_labels = [torch.tensor(SiameseDataset.target_class).long()] * len(self.target_paths)
            num_samples_per_writer = 1+len(self.target_paths)//len(self.chosen_ids)

            for label, wid in enumerate(list(data.keys())):
                if len(self.general_paths) >= len(self.target_paths):
                    break
                for i, p in enumerate(data[wid]):
                    if i >= num_samples_per_writer:
                        break
                    self.general_paths.append(p)
                    self.general_labels.append(torch.tensor(SiameseDataset.iam_class).long())

            self.target_paths += self.target_paths
            self.target_labels += self.target_labels
            self.general_paths += self.target_paths
            self.general_labels += self.target_labels



    def init_from_gt(self, gt_path, target_words_dir, iam_words_dir):
        self.typ1 = glob(os.path.join(iam_words_dir, '*'))[0].split('.')[-1]
        self.typ2 = glob(os.path.join(target_words_dir, '*'))[0].split('.')[-1]

        if str(len(balanced_classes)) in gt_path:
            self.chosen_ids = [label2wid[labels88_2labels500[lbl]] for lbl in balanced_classes]
        elif len(balanced_classes) == IAM_NUM_CLASSES:
            self.chosen_ids = [label2wid[lbl] for lbl in balanced_classes]
        assert self.chosen_ids is not None, "Invalid gt_path"

        self.target_paths = glob(os.path.join(target_words_dir, '*'))
        self.target_labels = [torch.tensor(SiameseDataset.target_class).long()] * len(self.target_paths)
        num_samples_per_writer = len(self.target_paths)//len(self.chosen_ids)

        with open(gt_path, 'r') as f:
            gt_lines = f.readlines()
            assert len(gt_lines) > 0, "Invalid gt file"

            # Add "IAM class" samples
            for wid in self.chosen_ids:
                if len(self.general_paths) >= len(self.target_paths):
                    break
                for i, line in enumerate(gt_lines):
                    if i >= num_samples_per_writer:
                        break
                    if int(line.split(',')[0]) == wid:
                        self.general_paths.append(os.path.join(iam_words_dir, f"{line.split(',')[-1].split(' ')[0]}.{self.typ1}"))
                        self.general_labels.append(torch.tensor(SiameseDataset.iam_class).long())
            
            self.target_paths += self.target_paths
            self.target_labels += self.target_labels
            self.general_paths += self.target_paths
            self.general_labels += self.target_labels


    def shuffle_pairs(self, paths, labels, start_ratio, end_ratio):
        paths_labels = list(zip(paths.copy(), labels.copy()))
        random.shuffle(paths_labels)
        new_paths, new_labels = zip(*paths_labels)
        new_paths = paths[int(len(paths)*start_ratio):int(len(paths)*end_ratio)]
        new_labels = labels[int(len(labels)*start_ratio):int(len(labels)*end_ratio)]
        return paths, labels


    def __getitem__(self, idx):
        target_x = cv2.imread(self.target_paths[idx], 0)
        general_idx = idx * random.randint(0, 100) % len(self.general_paths)
        general_x = cv2.imread(self.general_paths[general_idx], 0)

        if self.mode is modes[0]:
            target_x = train_preprocess(target_x, self.width, self.height)
            general_x = train_preprocess(general_x, self.width, self.height)
        else:
            target_x = test_preprocess(target_x, self.width, self.height)
            general_x = test_preprocess(general_x, self.width, self.height)

        target_x = torch.from_numpy(target_x).unsqueeze(0).float()
        general_x = torch.from_numpy(general_x).unsqueeze(0).float()
        
        general_y = self.general_labels[general_idx]
        target_y = self.target_labels[idx]
        return target_x, general_x, target_y, general_y


    def __len__(self):
        return len(self.target_paths)



class SiameseDataset2(D.Dataset):
    """
    json_path:          Path to the ground truth json file that contains the writer IDs, and the word images paths
    mode:               One of ['trian', 'val', 'test']
    width, height:      Images shape
    mode_start_ratio and mode_end_ratio: To determine where the mode begins and ends
    (e.g: 0.8. 1.0 --> Only the last 20% of the dataset paths are taken into consideration in this mode)
    """

    different_style = 0
    same_style = 1

    def __init__(self, json_path, mode, mode_start_ratio, mode_end_ratio, width, height):
        super(SiameseDataset2, self).__init__()
        global modes, balanced_classes, IAM_NUM_CLASSES

        assert json_path is not None, "json_path cannot be None"
        assert mode_start_ratio < mode_end_ratio and mode_start_ratio >= 0, "Invalid mode ratios"
        assert mode in modes, "Invalid mode"

        self.init_from_json(json_path)
        self.paths = self.paths[int(len(self.paths)*mode_start_ratio):int(len(self.paths)*mode_end_ratio)]
        self.width = width
        self.height = height
        self.mode = mode


    @staticmethod
    def dict_of_lists_to_list(d, target_key=None):
        l = list()
        keys = list(d.keys())
        for k in keys:
            if target_key != k:
                l += d[k]
        return l

    
    def choose_different_wid_paths(self, target_wid):
        wids = list(self.paths_by_wids.keys())
        found = False
        paths = None
        while not found:
            wid = random.sample(wids, 1)[0]
            if wid != target_wid:
                paths = self.paths_by_wids.get(wid)
                found = True
        return paths
          

    def init_from_json(self, json_path):
        self.length = 0
        self.paths = list()
        with open(json_path, 'r') as f:
            self.paths_by_wids = json.load(f)
            self.paths = SiameseDataset2.dict_of_lists_to_list(self.paths_by_wids)
        random.shuffle(self.paths)


    def __getitem__(self, idx):
        target_path = self.paths[idx]
        splitted1 = target_path.split('/')[-1].split('.')[0].split('-')
        text = ''.join(splitted1[1:-1])
        target_wid = str(int(splitted1[0]))

        paths_of_wid_with_same_text = [p for p in self.paths_by_wids.get(target_wid) if \
            ''.join(p.split('/')[-1].split('.')[0].split('-')[1:-1]) == text]
        same_style_path = random.sample(paths_of_wid_with_same_text, 1)[0]

        paths_of_different_wid_of_same_text = list()
        while len(paths_of_different_wid_of_same_text) == 0:
            paths_of_different_wid = self.choose_different_wid_paths(target_wid)
            paths_of_different_wid_of_same_text = [p for p in paths_of_different_wid if text == \
                ''.join(p.split('/')[-1].split('.')[0].split('-')[1:-1])]
        different_style_path = random.sample(paths_of_different_wid_of_same_text, 1)[0]

        target_x = cv2.imread(target_path, 0)
        same_style_x = cv2.imread(same_style_path, 0)
        different_style_x = cv2.imread(different_style_path, 0)
       
        if self.mode is modes[0]:
            target_x = train_preprocess(target_x, self.width, self.height)
            same_style_x = train_preprocess(same_style_x, self.width, self.height)
            different_style_x = train_preprocess(different_style_x, self.width, self.height)
        else:
            target_x = test_preprocess(target_x, self.width, self.height)
            same_style_x = test_preprocess(same_style_x, self.width, self.height)
            different_style_x = test_preprocess(different_style_x, self.width, self.height)

        target_x = torch.from_numpy(target_x).unsqueeze(0).float()
        same_style_x = torch.from_numpy(same_style_x).unsqueeze(0).float()
        different_style_x = torch.from_numpy(different_style_x).unsqueeze(0).float()
        
        return target_x, same_style_x, different_style_x, \
            torch.tensor(SiameseDataset2.same_style).long(), torch.tensor(SiameseDataset2.different_style).long()


    def __len__(self):
        return len(self.paths)


class Classic_DS(D.Dataset):
    def __init__(self, paths, labels):
        super(Classic_DS, self).__init__()
        self.paths = paths
        self.labels = labels

    def __getitem__(self, idx):
        x = cv2.imread(self.paths[idx], 0)
        x = test_preprocess(x)
        x = torch.from_numpy(x).unsqueeze(0).float()
        y = self.labels[idx]
        return x, y

    def __len__(self):
        return len(self.paths)


class Classic_Siamese_DS(D.Dataset):
    def __init__(self, target_paths, infer_paths, infer_labels):
        super(Classic_Siamese_DS, self).__init__()
        self.target_paths = target_paths
        self.infer_paths = infer_paths
        self.infer_labels = infer_labels


    def __getitem__(self, idx):
        global TARGET_CLASS

        tidx = random.randint(0, len(self.target_paths)-1)
        xt = cv2.imread(self.target_paths[tidx], 0)
        xt = test_preprocess(xt)
        xt = torch.from_numpy(xt).unsqueeze(0).float()

        xi = cv2.imread(self.infer_paths[idx], 0)
        xi = test_preprocess(xi)
        xi = torch.from_numpy(xi).unsqueeze(0).float()
        yi = self.infer_labels[idx]

        return xt, xi, TARGET_CLASS, yi


    def __len__(self):
        return len(self.infer_paths)

