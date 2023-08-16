import os
import torch
from torch.utils.data import Dataset
import json
from PIL import Image
from pathlib import Path
from .augment import create_AugTransforms
from torch.utils.data import DataLoader

class Datasets(Dataset):
    def __init__(self, root, mode, transforms = None, label_transforms = None, project = None, rank = None):
        assert os.path.isdir(root), "dataset root: {} does not exist.".format(root)
        src_path = os.path.join(root, mode)
        data_class = [cla for cla in os.listdir(src_path) if os.path.isdir(os.path.join(src_path, cla))]
        # sort
        data_class.sort()

        class_indices = dict((k, v) for v, k in enumerate(data_class))
        if rank in {-1, 0}:
            json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
            os.makedirs('./run', exist_ok=True)
            if project is not None:
                with open(Path(project) / 'class_indices.json', 'w') as json_file:
                    json_file.write(json_str)

        support = [".jpg", ".png"]

        images_path = []  # image path
        images_label = []  # label idx

        for cla in data_class:
            cla_path = os.path.join(src_path, cla)
            images = [os.path.join(src_path, cla, i) for i in os.listdir(cla_path)
                      if os.path.splitext(i)[-1] in support]
            image_class = class_indices[cla]
            for img_path in images:
                images_path.append(img_path)
                images_label.append(image_class)

        self.images = images_path
        self.labels = images_label
        self.transforms = transforms
        self.label_transforms = label_transforms
        self.class_indices = data_class
        self.mode = mode


    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transforms is not None:
            img = self.transforms(img)
        if self.label_transforms is not None:
            label = self.label_transforms(label)

        return img, label


    def __len__(self):
        return len(self.images)

    @staticmethod
    def collate_fn(batch):
        imgs, labels = tuple(zip(*batch))

        imgs = torch.stack(imgs, dim=0)
        labels = torch.as_tensor(labels) if isinstance(labels[0], int) else torch.stack(labels, dim=0)
        return imgs, labels

class MyDatasets:
    def __init__(self, data_cfgs: dict, rank, project):
        self.data_cfgs = data_cfgs # root, nw, imgsz, train, val
        self.rank = rank
        self.project = project
        self.label_transforms = None

        self.train_dataset = self.create_dataset('train')
        self.val_dataset = self.create_dataset('val')

    def create_dataset(self, mode: str):
        assert mode in {'train', 'val'}

        cfg = self.data_cfgs.get(mode, -1)
        if isinstance(cfg, dict):
            dataset = Datasets(root=self.data_cfgs['root'], mode=mode,
                               transforms=create_AugTransforms(augments=cfg['augment'], imgsz=self.data_cfgs['imgsz']),
                               project=self.project, rank=self.rank)
        else: dataset = None
        return dataset

    def set_augment(self, mode: str, sequence = None): # sequence -> T.Compose([...])
        if sequence is None:
            sequence = self.val_dataset.transforms
        dataset = getattr(self, f'{mode}_dataset')
        dataset.transforms = sequence

    def auto_aug_weaken(self, epoch: int, milestone: int):
        if epoch == milestone:
            self.set_augment('train', sequence = None)

    @staticmethod
    def set_label_transforms(label, num_classes, label_smooth): # idx -> vector
        vector = torch.zeros(num_classes).fill_(0.5 * label_smooth)
        vector[label] = 1 - 0.5 * label_smooth

        return vector

    @staticmethod
    def set_dataloader(dataset, bs: int = 256, nw: int = 0, pin_memory: bool = True, shuffle: bool = True, sampler = None):
        assert not (shuffle and sampler is not None)
        nd = torch.cuda.device_count()
        nw = min([os.cpu_count() // max(nd, 1), nw])
        return DataLoader(dataset=dataset, batch_size=bs, num_workers=nw, pin_memory=pin_memory, sampler=sampler, shuffle=shuffle)
