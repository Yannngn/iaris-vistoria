import torch
import torchmetrics
import torchvision

from torch import nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from scripts import transforms as T

def get_optimizer(params, hyperparams):
    assert hyperparams['optimizer'] in ['sgd', 'adam', None], f'{hyperparams["optimizer"]} not recognized'
    
    if hyperparams['optimizer'] == 'sgd': return torch.optim.SGD(params, lr=hyperparams['learning_rate'], momentum=0.9, weight_decay=0.0005)
    elif hyperparams['optimizer'] == 'adam': return torch.optim.Adam(params, lr=hyperparams['learning_rate'])
    else: pass

def get_criterion(hyperparams):
    assert hyperparams['criterion'] in ['crossentropy', None], f'{hyperparams["criterion"]} not recognized'
    
    if hyperparams["criterion"] == 'crossentropy': return nn.CrossEntropyLoss()
    else: pass

def get_scheduler(optimizer, hyperparams):
    assert hyperparams['scheduler'] in ['step', 'plateau', 'cyclic', None], f'{hyperparams["scheduler"]} not recognized'

    if hyperparams['scheduler'] == 'step': return torch.optim.lr_scheduler.StepLR(optimizer, step_size=hyperparams['scheduler_step'], gamma=0.1)
    elif hyperparams['scheduler'] == 'plateau': return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', step_size=hyperparams['scheduler_step'])
    elif hyperparams['scheduler'] == 'cyclic': return torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=hyperparams['learning_rate'] * 0.1, max_lr=hyperparams['learning_rate'], step_size_up=hyperparams['scheduler_step'], verbose=True)
    else: pass
    
def get_model_instance_classification(hyperparams):
    model = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.DEFAULT)
     
    model.classifier[2] = nn.Linear(4096, hyperparams['layer_2_size'])
    model.classifier[4] = nn.Linear(hyperparams['layer_2_size'], hyperparams['layer_4_size'])
    model.classifier[6] = nn.Linear(hyperparams['layer_4_size'], hyperparams['num_classes'])
    
    transform_train = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.Resize((hyperparams['image_size'], hyperparams['image_size'])),
                                                      torchvision.transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
                                                      # torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
                                                      torchvision.transforms.RandomHorizontalFlip(.5),
                                                      torchvision.transforms.RandomRotation(5)
                                                      ])

    transform_val = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Resize((hyperparams['image_size'], hyperparams['image_size'])),
                                                    torchvision.transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
                                                    ])

    return model, transform_train, transform_val

def get_model_instance_detection(hyperparams):
    def get_detection_transform(train):
        transforms = []
        transforms.append(T.PILToTensor())
        #transforms.append(T.Resize((hyperparams['image_size'], hyperparams['image_size'])))
        transforms.append(T.ConvertImageDtype(torch.float32))
        
        if train:
            transforms.append(T.RandomHorizontalFlip(0.5))
        return T.Compose(transforms)
 
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, hyperparams['num_classes'])

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    #hidden_layer = 256

    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hyperparams['hidden_layer'],
                                                       hyperparams['num_classes'])
    
    return model, get_detection_transform(True), get_detection_transform(False)

def get_metrics(config):
    num_classes = config['num_classes']
    average = 'weighted' if config['weighted'] else 'micro'

    device = torch.device('cpu')
    if config['device'] == 'gpu':
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    global_metrics = [
        torchmetrics.Accuracy(num_classes=num_classes, average=average, mdmc_average='global').to(device),
        torchmetrics.F1Score(num_classes=num_classes, average=average, mdmc_average='global').to(device),
        torchmetrics.Precision(num_classes=num_classes, average=average, mdmc_average='global').to(device),
        torchmetrics.Recall(num_classes=num_classes, average=average, mdmc_average='global').to(device),
        torchmetrics.Specificity(num_classes=num_classes, average=average, mdmc_average='global').to(device),
        torchmetrics.JaccardIndex(num_classes=num_classes, average=average, mdmc_average='global').to(device),
        torchmetrics.StatScores(num_classes=num_classes, average=average, mdmc_average='global').to(device)
    ]

    global_metrics_names = ["weighted accuracy", "f1", "precision", "recall", "specificity", 'jaccard', 'stats']

    label_metrics = [
        torchmetrics.Accuracy(num_classes=num_classes, average=None, mdmc_average='global').to(device),
        torchmetrics.F1Score(num_classes=num_classes, average=None, mdmc_average='global').to(device),
        torchmetrics.Precision(num_classes=num_classes, average=None, mdmc_average='global').to(device),
        torchmetrics.Recall(num_classes=num_classes, average=None, mdmc_average='global').to(device),
        torchmetrics.Specificity(num_classes=num_classes, average=None, mdmc_average='global').to(device),
        torchmetrics.JaccardIndex(num_classes=num_classes, average=None, mdmc_average='global').to(device),
        #torchmetrics.StatScores(num_classes=num_classes, average=None, mdmc_average='global').to(device)
    ]
      
    label_metrics_names = ["accuracy", "f1", "precision", "recall", "specificity", 'jaccard'] #'stats']
    
    global_dict = dict(zip(global_metrics_names, global_metrics))
    label_dict = dict(zip(label_metrics_names, label_metrics))

    return global_dict, label_dict

class ConfusionMatrixCallbackReuseImages():
    def __init__(self, model, experiment, inputs, targets, confusion_matrix):
        self.model = model
        self.experiment = experiment
        self.inputs = inputs
        self.targets = targets
        self.confusion_matrix = confusion_matrix

    def on_epoch_end(self, epoch, hyperparams, device):
        predicted = self.model(self.inputs.to(device))
        self.confusion_matrix.compute_matrix(self.targets.data.cpu().numpy(), 
                                             predicted.data.cpu().numpy(), 
                                             images=self.inputs.data.cpu().numpy(), 
                                             image_channels='first')
        
        self.experiment.log_confusion_matrix(
            matrix=self.confusion_matrix,
            title=f"Confusion Matrix, Epoch {epoch + 1}",
            file_name=f"confusion-matrix-{epoch + 1}.json", labels=hyperparams['labels']
        )