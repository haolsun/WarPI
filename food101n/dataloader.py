import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import os
from PIL import Image
import PIL


def resize(img, size, max_size=1000):
    '''Resize the input PIL image to the given size.
    Args:
      img: (PIL.Image) image to be resized.
      size: (tuple or int)
        - if is tuple, resize image to the size.
        - if is int, resize the shorter side to the size while maintaining the aspect ratio.
      max_size: (int) when size is int, limit the image longer size to max_size.
                This is essential to limit the usage of GPU memory.
    Returns:
      img: (PIL.Image) resized image.
    '''
    w, h = img.size
    if isinstance(size, int):
        size_min = min(w, h)
        sw = sh = float(size) / size_min

        ow = int(w * sw + 0.5)
        oh = int(h * sh + 0.5)
    else:
        ow, oh = size
        sw = float(ow) / w
        sh = float(oh) / h
    return img.resize((ow, oh), Image.BICUBIC)


class food101n(Dataset):
    def __init__(self, split='train', data_path=None, transform=None):
        if data_path is None:
            data_path = 'image_list'

        if split == 'train':
            self.image_list = np.load(os.path.join(data_path, 'train_images.npy'))
            self.targets = np.load(os.path.join(data_path, 'train_targets.npy'))
        elif split == 'meta':
            self.image_list = np.load(os.path.join(data_path, 'meta_images.npy'))
            self.targets = np.load(os.path.join(data_path, 'meta_targets.npy'))
        else:
            self.image_list = np.load(os.path.join(data_path, 'test_images.npy'))
            self.targets = np.load(os.path.join(data_path, 'test_targets.npy'))

        self.targets = self.targets  # make sure the label is in the range [0, num_class - 1]
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.image_list[index]
        image = Image.open(image_path)
        # image = image.resize((256, 256), resample=PIL.Image.BICUBIC)
        image = resize(image, 256)

        if self.transform is not None:
            image = self.transform(image)

        label = self.targets[index]
        label = np.array(label).astype(np.int64)

        return image, torch.from_numpy(label)

    def __len__(self):
        return len(self.targets)


