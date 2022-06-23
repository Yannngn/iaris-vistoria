import torch
import torchvision

from torch import nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from scripts import transforms as T

def get_model_instance_classification(hyperparams):
    model = torchvision.models.alexnet(pretrained=True)
    model.classifier[2] = nn.Linear(4096, hyperparams['layer_2_size'])
    model.classifier[4] = nn.Linear(hyperparams['layer_2_size'], hyperparams['layer_4_size'])
    model.classifier[6] = nn.Linear(hyperparams['layer_4_size'], hyperparams['num_classes'])
    
    transform_train = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.Resize((hyperparams['image_size'], hyperparams['image_size'])),
                                                      torchvision.transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
                                                      #torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
                                                      torchvision.transforms.RandomHorizontalFlip(.5),
                                                      torchvision.transforms.RandomRotation(5)
                                                      ])

    transform_val = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Resize((hyperparams['image_size'], hyperparams['image_size'])),
                                                    torchvision.transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
                                                    ])

    return model, transform_train, transform_val

def get_model_instance_detection(hyperparams):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, hyperparams['num_classes'])

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    #hidden_layer = 256

    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hyperparams['hidden_layer'],
                                                       hyperparams['num_classes'])
    
    return model, get_detection_transform(True), get_detection_transform(False)

def get_detection_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float32))
    
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)