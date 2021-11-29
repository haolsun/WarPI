from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import random
import os.path


class Clothing1M(Dataset):
    def __init__(self, root, mode='train', transform=None):

        self.transform = transform
        self.noise_data = {}
        self.clean_data = {}
        self.datas = []
        self.labels = []

        if mode == 'train':
            with open('data/noisy_label_kv.txt', 'r') as f:
                lines = f.readlines()
                for l in lines:
                    detail = l.split()
                    img_path = os.path.join(root, detail[0])
                    self.noise_data[img_path] = int(detail[1])
        else:
            with open('data/clean_label_kv.txt', 'r') as f:
                lines = f.readlines()
                for l in lines:
                    detail = l.split()
                    img_path = os.path.join(root, detail[0])
                    self.clean_data[img_path] = int(detail[1])

        if mode == 'train':
            with open('data/noisy_train_key_list.txt', 'r') as f:
                lines = f.readlines()
                random.shuffle(lines)
                for l in lines:
                    s = l.split('.')
                    l = s[0] + '.jpg'
                    img_path = os.path.join(root, l)
                    label = self.noise_data[img_path]

                    self.datas.append(img_path)
                    self.labels.append(label)

        elif mode == 'val':
            with open('data/clean_val_key_list.txt', 'r') as f:
                lines = f.readlines()
                random.shuffle(lines)
                for l in lines:
                    s = l.split('.')
                    l = s[0] + '.jpg'
                    img_path = os.path.join(root, l)
                    label = self.clean_data[img_path]

                    self.datas.append(img_path)
                    self.labels.append(label)

        elif mode == 'test':
            with open('data/clean_test_key_list.txt', 'r') as f:
                lines = f.readlines()
                random.shuffle(lines)
                for l in lines:
                    s = l.split('.')
                    l = s[0] + '.jpg'
                    img_path = os.path.join(root, l)
                    label = self.clean_data[img_path]

                    self.datas.append(img_path)
                    self.labels.append(label)


    def __getitem__(self, index):
        impath, target = self.datas[index], self.labels[index]
        img = Image.open(impath).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.datas)



