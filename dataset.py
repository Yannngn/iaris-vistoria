import numpy as np
import torch

from PIL import Image

from scripts import utils as SU

def get_data(images, masks, train_transform, test_transform, hyperparams):
    if hyperparams['type'] == 'detection':
        dataset = DetectionDataset(images, masks, train_transform)
        dataset_test = DetectionDataset(images, masks, test_transform)
    
    elif hyperparams['type'] == 'classification':
        dataset = ClassificationDataset(images, masks, train_transform)
        dataset_test = ClassificationDataset(images, masks, test_transform)

    indices = torch.randperm(len(dataset)).tolist()
    length = int(.7 * len(dataset))
    
    dataset = torch.utils.data.Subset(dataset, indices[:length])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[length:])
    
    train = torch.utils.data.DataLoader(dataset, batch_size=hyperparams['batch_size'], 
                                        shuffle=True, num_workers=2, 
                                        collate_fn=SU.collate_fn)

    test = torch.utils.data.DataLoader(dataset_test, batch_size=hyperparams['batch_size'], 
                                        shuffle=False, num_workers=2, 
                                        collate_fn=SU.collate_fn)       
    
    return train, test

class Dataset(torch.utils.data.Dataset):
    def __init__(self, images, target, transforms):
        self.transforms = transforms
        self.images = images
        self.labels = target
        pass
    
    def __getitem__(self):
        pass
    
    def __len__(self):
        return len(self.images)
    
class ClassificationDataset(Dataset):
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transforms is not None:
            img = self.transforms(img)

        return img, label
    
class DetectionDataset(Dataset):
    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]
        img = Image.open(img_path).convert("RGB")

        mask = Image.open(mask_path)
        mask = np.array(mask)
        
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]
        masks = mask == obj_ids[:, None, None]

        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            boxes.append([np.min(pos[1]), np.min(pos[0]), np.max(pos[1]), np.max(pos[0])])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {"boxes": boxes,
                  "labels": labels,
                  "masks": masks,
                  "image_id": image_id,
                  "area": area,
                  "iscrowd": iscrowd}
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target