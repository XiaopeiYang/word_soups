import os
import json
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
#import imp_samp
import pickle
import math

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import read_json, write_json, listdir_nohidden, mkdir_if_missing

#from .oxford_pets import OxfordPets
#from .dtd import DescribableTextures as DTD
import argparse
from argparse_parameters import get_arg_parser

#parser = get_arg_parser()
#args = parser.parse_args()
#print("fungismallarg>",args)

NEW_CNAMES = {
    "CARB": "Carbendazim",
    "CASPO": "Caspofungin",
    "GWT1": "GWT1",
    "GermTube": "GermTube",
    "MYCELIUM": "Mycelium",
    "TEBUCO": "Tebuconazole",
}

@DATASET_REGISTRY.register()
class FungiSmall(DatasetBase):

    dataset_dir = "images4LMU"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        #self.image_dir = os.path.join(self.dataset_dir, "images")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_FungiSmall.json")
        self.split_fewshot_dir = os.path.join(os.getcwd(), "datasets/splits/fungismall")
        mkdir_if_missing(self.split_fewshot_dir)
        self.use_patches = cfg.USE_PATCHES
        #self.use_patches = True

        # Decide whether to use patches based on the configuration
        #use_patches = cfg.get("USE_PATCHES", True)

        if os.path.exists(self.split_path):
            train, val, test = self.read_split(self.split_path, self.dataset_dir)
        else:
            # Pass the use_patches flag and imp_samp_params to the function
            train, val, test = self.read_and_split_data(self.dataset_dir, new_cnames=NEW_CNAMES)
            self.save_split(train, val, test, self.split_path, self.dataset_dir)

        num_shots = cfg.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.SUBSAMPLE_CLASSES
        train, val, test = self.subsample_classes(train, val, test, subsample=subsample)
        train, val, test = self.fix_image_paths([train, val, test], "images4LMU", root)

        super().__init__(train_x=train, val=val, test=test)

    @staticmethod
    def read_and_split_data(dataset_dir, p_trn=0.5, p_val=0.2, ignored=[], new_cnames=None):
        # The data are supposed to be organized into the following structure
        # =============
        # images4LMU/
        #     CARB/
        #     CASPO/
        #     GWT1/
        #     GermTube/
        #     MYCELIUM/
        #     TEBUCO/
        # =============
        categories = listdir_nohidden(dataset_dir)
        categories = [c for c in categories if c not in ignored]
        categories.sort()

        p_tst = 1 - p_trn - p_val
        print(f"Splitting into {p_trn:.0%} train, {p_val:.0%} val, and {p_tst:.0%} test")

        def _collate(ims, y, c):
            items = []
            for im in ims:
                item = Datum(impath=im, label=y, classname=c)  # is already 0-based
                items.append(item)
            return items

        train, val, test = [], [], []
        for label, category in enumerate(categories):
            if os.path.isdir(os.path.join(dataset_dir, category)):
                category_dir = os.path.join(dataset_dir, category)
            images = listdir_nohidden(category_dir)
            images = [os.path.join(category_dir, im) for im in images]
            random.shuffle(images)
            n_total = len(images)
            n_train = round(n_total * p_trn)
            n_val = round(n_total * p_val)
            n_test = n_total - n_train - n_val
            assert n_train > 0 and n_val > 0 and n_test > 0

            if new_cnames is not None and category in new_cnames:
                category = new_cnames[category]

            train.extend(_collate(images[:n_train], label, category))
            val.extend(_collate(images[n_train : n_train + n_val], label, category))
            test.extend(_collate(images[n_train + n_val :], label, category))

        return train, val, test
    
    @staticmethod
    def read_split(filepath, path_prefix):
        def _convert(items):
            out = []
            for impath, label, classname in items:
                impath = os.path.join(path_prefix, impath)
                item = Datum(impath=impath, label=int(label), classname=classname)
                out.append(item)
            return out

        print(f"Reading split from {filepath}")
        split = read_json(filepath)
        train = _convert(split["train"])
        val = _convert(split["val"])
        test = _convert(split["test"])

        return train, val, test
    
    @staticmethod
    def save_split(train, val, test, filepath, path_prefix):
        def _extract(items):
            out = []
            for item in items:
                impath = item.impath
                label = item.label
                classname = item.classname
                impath = impath.replace(path_prefix, "")
                if impath.startswith("/"):
                    impath = impath[1:]
                out.append((impath, label, classname))
            return out

        train = _extract(train)
        val = _extract(val)
        test = _extract(test)

        split = {"train": train, "val": val, "test": test}

        write_json(split, filepath)
        print(f"Saved split to {filepath}")

    @staticmethod
    def subsample_classes(*args, subsample="all"):
        """Divide classes into two groups. The first group
        represents base classes while the second group represents
        new classes.

        Args:
            args: a list of datasets, e.g. train, val and test.
            subsample (str): what classes to subsample.
        """
        assert subsample in ["all", "base", "new"]

        if subsample == "all":
            return args
        
        dataset = args[0]
        labels = set()
        for item in dataset:
            labels.add(item.label)
        labels = list(labels)
        labels.sort()
        n = len(labels)
        # Divide classes into two halves
        m = math.ceil(n / 2)

        print(f"SUBSAMPLE {subsample.upper()} CLASSES!")
        if subsample == "base":
            selected = labels[:m]  # take the first half
        else:
            selected = labels[m:]  # take the second half
        relabeler = {y: y_new for y_new, y in enumerate(selected)}
        
        output = []
        for dataset in args:
            dataset_new = []
            for item in dataset:
                if item.label not in selected:
                    continue
                item_new = Datum(
                    impath=item.impath,
                    label=relabeler[item.label],
                    classname=item.classname
                )
                dataset_new.append(item_new)
            output.append(dataset_new)
        
        return output

    @staticmethod
    def fix_image_paths(dsets, dataset_dir, root):
        '''
        Notes:

        type(dset) can be anything, e.g. datasets.imagenet.ImageNet
        dset.train_x and dset.test are lists containing objexts of type
        dassl.data.datasets.base_dataset.Datum
        each Datum has attributes label and impath indicating the label 
        and absolute path the image file, respectively.

        These splits are pregenerated and saved in the folder 
        ``datasets/splits/<dataset name>/*.pkl``
        '''

        out = []
        for dset in dsets:
            dset_new = []
            for item in dset:
                new_impath = os.path.join(root, dataset_dir, item.impath)
                item_new = Datum(
                    impath=new_impath,
                    label=item.label,
                    classname=item.classname
                )
                dset_new.append(item_new)
            out.append(dset_new)
        return out

#parser = get_arg_parser()
#args = parser.parse_args()
#cfg = argparse.Namespace()
#cfg.ROOT = args.data_dir
#cfg.NUM_SHOTS = 16
#cfg.SEED = 1
#cfg.SUBSAMPLE_CLASSES = 'all'
#dset = FungiSmall(cfg)