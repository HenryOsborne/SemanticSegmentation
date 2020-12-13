import os
import numpy as np
from torchvision import transforms
import torch.utils.data.dataloader as dataloader
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from utils import custom_transforms as tr


class ADE20K(Dataset):
    NUM_Class = 150

    def __init__(self, args, base_dir='dataset/ade20k', split='train'):
        super(ADE20K, self).__init__()
        self.args = args
        self.base_dir = base_dir
        self.split_dir = split
        self.img_dir = os.path.join(self.base_dir, self.split_dir, 'image')
        self.anno_dir = os.path.join(self.base_dir, self.split_dir, 'label')

        img_list = [i.split('.')[0] for i in os.listdir(self.img_dir)]

        self.im_ids = []
        self.images = []
        self.labels = []

        for i, img in enumerate(img_list):
            image = os.path.join(self.img_dir, img + '.jpg')
            anno = os.path.join(self.anno_dir, img + '.png')
            assert os.path.isfile(image)
            assert os.path.isfile(anno)
            self.im_ids.append(img)
            self.images.append(image)
            self.labels.append(anno)

        assert len(self.images) == len(self.labels)
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img = Image.open(self.images[item]).convert('RGB')
        target = Image.open(self.labels[item])

        if self.split_dir == 'train':
            pass
        elif self.split_dir == 'val':
            pass
        else:
            raise NotImplementedError

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)
