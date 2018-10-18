from __future__ import print_function

from typing import Optional, List, Any, Union, Tuple

from PIL import Image
from os.path import join
import os
import random
import torch
import torch.utils.data as data
import numpy as np
from .utils import download_url, check_integrity, list_dir, list_files


class Omniglot(data.Dataset):
    """`Omniglot <https://github.com/brendenlake/omniglot>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``omniglot-py`` exists.
        background (bool, optional): If True, creates dataset from the "background" set, otherwise
            creates from the "evaluation" set. This terminology is defined by the authors.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset zip files from the internet and
            puts it in root directory. If the zip files are already downloaded, they are not
            downloaded again.
    """
    folder = 'omniglot-py'
    download_url_prefix = 'https://github.com/brendenlake/omniglot/raw/master/python'
    zips_md5 = {
        'images_background': '68d2efa1b9178cc56df9314c21c6e718',
        'images_evaluation': '6b91aef0f799c5bb55b94e3f2daec811'
    }

    def __init__(self, root, background="tra",
                 transform=None, target_transform=None,
                 download=False):
        self.root = join(os.path.expanduser(root), self.folder)
        self.background = background
        self.transform = transform
        self.target_transform = target_transform
        if background == "tra":
            self.num_langs = 24
        elif background == "val":
            self.num_langs = 6
        else:
            self.num_langs = 20

        if download:
            self.download()

        # if not self._check_integrity():
        #     raise RuntimeError('Dataset not found or corrupted.' +
        #                        ' You can use download=True to download it')

        self.target_folder = join(self.root, self._get_target_folder())
        self._alphabets = list_dir(self.target_folder)

        self._characters = sum([[[join(a, c) for c in list_dir(join(self.target_folder, a))]
                                for a in self._alphabets]], [])
        # print(self._characters)
        # print(list_dir(join(self.root, self._get_target_folder())))
        self._character_images = []
        # for _ in range(len(list_dir(join(self.root, self._get_target_folder())))):
        counter = 0
        for curr in self._characters:
            self._character_images.append([[(image, idx, counter) for image in list_files(join(self.target_folder, character), '.png')]
                                           for idx, character in enumerate(curr)])
            counter += 1
        self._flat_character_images = []
        self._characters_backup = list_dir(self.target_folder)

        # for _ in range(self.num_langs):
        #     self._flat_character_images.append([])  # sum(self._character_images, [])
        self._flat_character_images = sum(sum(self._character_images, []), [])
        print(len(self._flat_character_images))
        # for outer in range(len(self._character_images)):
        #     for inner in range(len(self._character_images[outer])):
        #         self._flat_character_images[outer].extend(self._character_images[outer][inner])
        # print(self._character_images[0:50])
        # print("="*100)
        # print(self._flat_character_images[0:50])
        # print("=" * 100)
        self.pairs = self.generate_random_num_pairs()

    def generate_random_num_pairs(self):
        random.shuffle(self._flat_character_images)
        samp = random.sample(range(len(self._flat_character_images)), len(self._flat_character_images))
        to_return = ([(samp[j], samp[j + 1]) for j in range(len(samp) - 1)])

        return to_return

    def __len__(self):
        return len(self._flat_character_images)//2

    def get_rand_pair(self, index):  # Issue of using same image again
        pair = self.pairs[index]

        return [self._flat_character_images[pair[0]],
                self._flat_character_images[pair[1]]], pair

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target character class.
        """
        images = []
        target = 0
        image_rand_pairs = self.get_rand_pair(index)
        image_pair = image_rand_pairs[0]
        # print("="*5, index, "="*5)
        # print(image_pair)
        for i in range(0, 2):
            image_name, character_class, alpha = image_pair[i]
            # print(str(i) + " CHAR:", self.target_folder, self._characters_backup[alpha],
            #       "character{:02d}".format(self._flat_character_images[image_rand_pairs[1][i]][1]+1),
            #       image_name)
            image_path = join(self.target_folder, self._characters_backup[alpha],
                              "character{:02d}".format(
                                  self._flat_character_images[image_rand_pairs[1][i]][1] + 1),
                              image_name)
            image = Image.open(image_path, mode='r').convert('L')

            if self.transform:
                image = self.transform(image)

            images.append(image)

            if self.target_transform:
                character_class = self.target_transform(character_class)

        target = 1 if image_pair[0][2] == image_pair[1][2] else 0
        # print(torch.from_numpy(np.array(images), target)
        return images, torch.tensor(target, dtype=torch.float32)

    def _check_integrity(self):
        zip_filename = self._get_target_folder()
        if not check_integrity(join(self.root, zip_filename + '.zip'), self.zips_md5[zip_filename]):
            return False
        return True

    def download(self):
        import zipfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        filename = self._get_target_folder()
        zip_filename = filename + '.zip'
        url = self.download_url_prefix + '/' + zip_filename
        download_url(url, self.root, zip_filename, self.zips_md5[filename])
        print('Extracting downloaded file: ' + join(self.root, zip_filename))
        with zipfile.ZipFile(join(self.root, zip_filename), 'r') as zip_file:
            zip_file.extractall(self.root)

    def _get_target_folder(self):
        if self.background == "tra":
            return 'images_background'
        elif self.background == "val":
            return 'images_validation'
        else:
            return 'images_evaluation'
