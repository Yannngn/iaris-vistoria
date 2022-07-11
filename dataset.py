
import numpy as np
import os
import pandas as pd
import shutil
import torch
import torchvision

from PIL import Image
from typing import Optional, Tuple, Union

from scripts import utils as SU

def get_images_and_labels_from_df(
        input: str, 
        output: Optional[str], 
        params: dict
        )-> Tuple[list, list]:

    ''' Returns images and labels paths, if copy_files is True it copies the files from the original location to the output
        Args:
            :param str input: path to input csv
            :param str output: new path for copying the images
            :param dict params: parameters dictionary with "type", "image_column" and "target", "classification"/"detection", column that contains the images path and column that contains the labels/masks
            :param bool copy_files: copy files to output if provided
        Returns: 
            :return: images and labels, lists with the paths
    '''
    
    df = pd.read_csv(input)    
    images = df[params['image_column']].values.tolist()
    if output:
        temp = os.path.dirname(images[-1])
        images = [img.replace(temp, os.path.join(output, 'images')).replace('\\', '/') for img in images]
        print(images[0])

             
    if params['type'] == 'classification':
        labels = df[params['target']].values.tolist()
        params['labels'] = list(set(df[f'{params["target"]}_label'])) 
        for i, v in enumerate(params['labels']):
            labels = [l if l != v else i for l in labels]

            tup = [(a,b) for a,b  in zip(images, labels) if b]
            images = [a for a, _ in tup]
            labels = [b for _, b in tup]

    elif params['type'] == 'detection':
        labels = df[f"{params['target']}_masks"].values.tolist()
        if output:
            temp = os.path.dirname(labels[-1])
            labels = [lab.replace(temp, os.path.join(output, 'combined_masks')) if type(lab) is str else None for lab in labels]
        ## Quase certeza que dá pra fazer isso numa unica comprehension, mas ValueError: too many values to unpack (expected 2)
        tup = [(a,b) for a,b  in zip(images, labels) if b]
        images = [a for a, _ in tup]
        labels = [b for _, b in tup]

    return images, labels

def get_images_from_drive(
        input: str, 
        output: str
        ) -> None:

    '''Copies files from input directory to output directory
    
    Args:
        :param str input: input directory
        :param str output: output directory
    
    '''
    files = os.listdir(input)
    for name in files:
        full_name = os.path.join(input, name)
        if not os.path.isfile(full_name): continue       
        shutil.copy(full_name, output)

def get_data(
        images: list, 
        labels: list, 
        train_transform: torchvision.transforms, 
        test_transform: torchvision.transforms, 
        hyperparams: dict
        ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    
    '''Returns train and test dataloaders, providing images, labels/masks, transforms and parameters\n

    Args:
        images (list): images list
        labels (list): labels list, mask paths or label integers
        train_transform (torchvision.transforms): transforms
        test_transform (torchvision.transforms): transforms
        hyperparams (dict): parameters dict

    Returns:
        Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]: train and test dataloaders.
    
    '''
    
    if hyperparams['type'] == 'detection':
        dataset = DetectionDataset(images, labels, train_transform)
        dataset_test = DetectionDataset(images, labels, test_transform)
    
    elif hyperparams['type'] == 'classification':
        dataset = ClassificationDataset(images, labels, train_transform)
        dataset_test = ClassificationDataset(images, labels, test_transform)

    indices = torch.randperm(len(dataset)).tolist()
    length = int(.7 * len(dataset))
    
    dataset = torch.utils.data.Subset(dataset, indices[:length])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[length:])
    
    if hyperparams['type'] == 'detection':
        train = torch.utils.data.DataLoader(dataset, batch_size=hyperparams['batch_size'], 
                                            shuffle=True, num_workers=0, 
                                            collate_fn=SU.collate_fn)

        test = torch.utils.data.DataLoader(dataset_test, batch_size=hyperparams['batch_size'], 
                                            shuffle=False, num_workers=0, 
                                            collate_fn=SU.collate_fn)
        
    elif hyperparams['type'] == 'classification':      
        train = torch.utils.data.DataLoader(dataset, batch_size=hyperparams['batch_size'], 
                                            shuffle=True, num_workers=0)

        test = torch.utils.data.DataLoader(dataset_test, batch_size=hyperparams['batch_size'], 
                                            shuffle=False, num_workers=0)
            
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
        mask_path = self.labels[idx]
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