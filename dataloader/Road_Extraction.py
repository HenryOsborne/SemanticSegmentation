from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from utils import custom_transforms as tr


class RoadSegmentation(Dataset):
    """
    PascalVoc dataset
    """
    NUM_CLASSES = 2

    def __init__(self,
                 args,
                 base_dir='dataset/DeepGlobe',
                 split='train',
                 ):
        """
        :param base_dir: path to DeepGlobe dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir, 'JPEGImages')
        self._cat_dir = os.path.join(self._base_dir, 'Annotations')

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.args = args

        _splits_dir = self._base_dir

        self.im_ids = []
        self.images = []
        self.categories = []
        # R_mean:104.47535231777452, G_mean:97.62286006787794, B_mean:73.47118156177608
        # R_std:39.84996037265292, G_std:32.301591901960364, B_std:31.26034185372447
        self.mean = (104.47535231777452 / 255, 97.62286006787794 / 255, 73.47118156177608 / 255)
        self.std = (39.84996037265292 / 255, 32.301591901960364 / 255, 31.26034185372447 / 255)

        for splt in self.split:
            with open(os.path.join(os.path.join(_splits_dir, splt + '.txt')), "r") as f:
                lines = f.read().splitlines()

            for ii, line in enumerate(lines):
                _image = os.path.join(self._image_dir, line + ".jpg")
                _cat = os.path.join(self._cat_dir, line + ".png")
                assert os.path.isfile(_image)
                assert os.path.isfile(_cat)
                self.im_ids.append(line)
                self.images.append(_image)
                self.categories.append(_cat)

        assert (len(self.images) == len(self.categories))

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}

        for split in self.split:
            if split == "train":
                return self.transform_tr(sample)
            elif split == 'val':
                return self.transform_val(sample)

    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _target = Image.open(self.categories[index])

        return _img, _target

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=self.mean, std=self.std),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=self.mean, std=self.std),
            tr.ToTensor()])

        return composed_transforms(sample)

    def __str__(self):
        return 'Road(split=' + str(self.split) + ')'


if __name__ == '__main__':
    from utils.get_label import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 512
    args.crop_size = 256

    road_train = RoadSegmentation(args, split='train')

    dataloader = DataLoader(road_train, batch_size=5, shuffle=True, num_workers=0)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='road')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (39.84996037265292 / 255, 32.301591901960364 / 255, 31.26034185372447 / 255)
            img_tmp += (104.47535231777452 / 255, 97.62286006787794 / 255, 73.47118156177608 / 255)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)
