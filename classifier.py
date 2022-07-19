import comet_ml
import numpy as np
import os
import torch
import torchvision

from tqdm import tqdm
from typing import Optional, Tuple, List

from dataset import get_data
from utils import get_model_instance_classification, get_criterion, get_optimizer, get_scheduler, get_metrics
from metrics import evaluate_classification

def train_classifier(
        train_data: Tuple(List, List), 
        test_data: Tuple(List, List), 
        hyperparams: dict, 
        comet: bool = True) -> None:
    
    """_summary_

    Args:
        images (list): _description_
        masks (list): _description_
        hyperparams (dict): _description_
        comet (bool, optional): _description_. Defaults to True.
    """    
    
    device = torch.device('cpu')
    if hyperparams['device'] == 'gpu':
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model, train_transform, test_transform = get_model_instance_classification(hyperparams)
    model.to(device)
    
    dataloader_train, dataloader_test = get_data(train_data, test_data, train_transform, test_transform, hyperparams)
    
    loss_criterion = get_criterion(hyperparams)
    optimizer = get_optimizer(model.parameters(), hyperparams)
    scheduler = get_scheduler(optimizer, hyperparams)
        
    if comet:
        comet_train_loop(model, loss_criterion, optimizer, scheduler, dataloader_train, dataloader_test, device, hyperparams)
    else:
        train_loop(model, loss_criterion, optimizer, scheduler, dataloader_train, dataloader_test, device, hyperparams)
    
    torch.save(model.state_dict(), f'models/{hyperparams["time"]}_{hyperparams["target"]}.pickle')

def train_loop(
        model, 
        loss_criterion, 
        optimizer, 
        scheduler, 
        dataloader_train, 
        dataloader_test, 
        device, 
        hyperparams: dict):
    
    """ Runs training loop for N epochs
    
    Args:
        model (torch.module model): Model to be trained
        loss_criterion (torch.nn loss function): Loss function 
        optimizer: (torch.optim optimizer): Optimizer  
        scheduler: (torch.optim.LR_Scheduler): Learning Rate Scheduler  
        dataloader_train: (torch.utils.data.DataLoader): Train dataloader
        dataloader_test: (torch.utils.data.DataLoader): Test dataloader
        device: (torch.device): device
        hyperparams (dict): parameters dict
    """    
    
    for epoch in range(hyperparams['num_epochs']):
        print(f'Starting training epoch {epoch}:')
        
        train_loss, train_accuracy = train(model, loss_criterion, optimizer, dataloader_train, device)
        
        print(f'Loss: {train_loss}; Accuracy: {train_accuracy}')
        print()
        print(f'Validating epoch {epoch}:')
               
        val_loss, val_accuracy = validate(model, loss_criterion, scheduler, dataloader_test, device)
        
        print(f'Loss: {val_loss}; Accuracy: {val_accuracy}')

def comet_train_loop(
        model, 
        loss_criterion, 
        optimizer, 
        scheduler, 
        dataloader_train, 
        dataloader_val, 
        device, 
        hyperparams: dict):
    
    """ Runs training loop for N epochs while logging to comet_ml
    
    Args:
        model (torch.module model): Model to be trained
        loss_criterion (torch.nn loss function): Loss function 
        optimizer: (torch.optim optimizer): Optimizer  
        scheduler: (torch.optim.LR_Scheduler): Learning Rate Scheduler  
        dataloader_train: (torch.utils.data.DataLoader): Train dataloader
        dataloader_test: (torch.utils.data.DataLoader): Test dataloader
        device: (torch.device): device
        hyperparams (dict): parameters dict
    """   

    comet_ml.init(api_key=hyperparams['comet_api_key'])
    experiment = comet_ml.Experiment(api_key=hyperparams['comet_api_key'], project_name=hyperparams['comet_project_name'])
    experiment.log_parameters(hyperparams)   
    callback_cm = confusion_matrix_comet(model, dataloader_val, experiment, hyperparams, device)
    global_metrics, label_metrics = get_metrics(hyperparams)
    
    for epoch in range(hyperparams['num_epochs']):
        experiment.set_epoch(epoch)
        experiment.set_step(epoch)
        print(f'Starting training epoch {epoch}:') 

        train_loss, train_accuracy = train(model, loss_criterion, optimizer, dataloader_train, device)
        with experiment.train():
            experiment.log_metrics({"accuracy": train_accuracy, "loss": train_loss})        
        print(f'TRAIN: \t Loss: {train_loss}; Accuracy: {train_accuracy}')
        print()
        
        if epoch % hyperparams['validation_interval'] == 0 or epoch == hyperparams['num_epochs'] - 1:
            print(f'Validating and logging epoch {epoch}:')
            val_loss, val_accuracy, metrics = expensive_validate(model, loss_criterion, scheduler, dataloader_val, global_metrics, label_metrics, hyperparams, device)
            
        else:
            print(f'Validating epoch {epoch}:')
            val_loss, val_accuracy = validate(model, loss_criterion, scheduler, dataloader_val, device)
            metrics = None
        
        with experiment.validate():
            if not metrics:
                metrics = {}  

            metrics['accuracy'] = val_accuracy
            metrics['loss'] = val_loss
            experiment.log_metrics(metrics)
        
        callback_cm.on_epoch_end(epoch, hyperparams, device)
        
        print(f'VALIDATION: \t Loss: {val_loss}; Accuracy: {val_accuracy}')            
        
        if epoch % hyperparams['checkpoint_interval'] == 0 or epoch == hyperparams['num_epochs'] - 1:
            torch.save(model.state_dict(), os.path.join(os.path.dirname(os.path.abspath(__file__)), f'models/{hyperparams["time"]}_{hyperparams["target"]}.pickle'))
            
        experiment.log_epoch_end(epoch)   
        
def train(
        model, 
        loss_criterion, 
        optimizer, 
        dataloader, 
        device) -> Tuple[float, float]:
    
    batch_accuracy, batch_loss = [], []
    model.train()
    loader = tqdm(dataloader)
    for _, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        
        loss = loss_criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(outputs.data, 1)
        batch_total = labels.size(0)
        batch_correct = (predicted == labels.data).cpu().sum().numpy()
        batch_accuracy.append(batch_correct / batch_total)
                        
        batch_loss.append(loss.item())
        
    epoch_loss = np.sum(batch_loss)
    epoch_accuracy = np.mean(batch_accuracy)
    
    return epoch_loss, epoch_accuracy

def validate(model, loss_criterion, scheduler, dataloader, device):
    batch_accuracy, batch_loss = [], []
    loader = tqdm(dataloader)
    with torch.no_grad():
        model.eval()
        for _, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            loss = loss_criterion(outputs, labels)
            scheduler.step()

            _, predicted = torch.max(outputs.data, 1)
            batch_total = labels.size(0)
            batch_correct = (predicted == labels.data).cpu().sum().numpy()
            batch_accuracy.append(batch_correct / batch_total)
                            
            batch_loss.append(loss.item())
    
        epoch_loss = np.sum(batch_loss)
        epoch_accuracy = np.mean(batch_accuracy)
          
    return epoch_loss, epoch_accuracy

def expensive_validate(model, loss_criterion, scheduler, dataloader, global_metrics, label_metrics, hyperparams, device):
    batch_accuracy, batch_loss = [], []
    loader = tqdm(dataloader)
    with torch.no_grad():
        model.eval()
        for i, (inputs, labels) in enumerate(loader):
            _dict = {}
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)

            loss = loss_criterion(outputs, labels)
            scheduler.step()

            _, predicted = torch.max(outputs.data, 1)
            batch_total = labels.size(0)
            batch_correct = (predicted == labels.data).cpu().sum().numpy()
            batch_accuracy.append(batch_correct / batch_total)
                            
            batch_loss.append(loss.item())

            _dict = evaluate_classification(outputs, labels, global_metrics, label_metrics, hyperparams)
#            _dict['step'] = epoch * len(dataloader) + i
            
            
        epoch_loss = np.sum(batch_loss)
        epoch_accuracy = np.mean(batch_accuracy)
          
    return epoch_loss, epoch_accuracy, _dict

def confusion_matrix_comet(model, dataloader, experiment, hyperparams, device):
    from utils import ConfusionMatrixCallbackReuseImages
    confusion_matrix = experiment.create_confusion_matrix()
    
    dataiter = iter(dataloader)
    x, y = dataiter.next()
    y_pred = model(x.to(device)) 
    
    confusion_matrix.compute_matrix(y.data.numpy(), y_pred.data.cpu().numpy(), images=x.data.numpy(), image_channels='first')
    experiment.log_confusion_matrix(matrix=confusion_matrix, step=0, title=f"Confusion Matrix, Epoch 0", file_name=f"confusion-matrix-0.json", labels=hyperparams['labels'])
    
    return ConfusionMatrixCallbackReuseImages(model, experiment, x, y, confusion_matrix)