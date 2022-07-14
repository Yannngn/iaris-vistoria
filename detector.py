import comet_ml
import torch

from datetime import datetime

from scripts.engine import train_one_epoch, evaluate

from dataset import get_data
from utils import get_model_instance_detection, get_optimizer, get_scheduler

def train_detector(images, masks, hyperparams, comet=True):
    device = torch.device('cpu')
    if hyperparams['device'] == 'gpu':
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model, train_transform, test_transform = get_model_instance_detection(hyperparams)
    model.to(device)
    
    dataloader_train, dataloader_test = get_data(images, masks, train_transform, test_transform, hyperparams)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = get_optimizer(params, hyperparams)
    scheduler = get_scheduler(optimizer, hyperparams)
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    hyperparams['time'] = now
    
    if 'print_freq' not in hyperparams: hyperparams['print_freq'] = 10
    
    if hyperparams['use_comet']:
        comet_train_loop(model, optimizer, scheduler, dataloader_train, dataloader_test, device, hyperparams)
    else:
        train_loop(model, optimizer, scheduler, dataloader_train, dataloader_test, device, hyperparams)
    
    torch.save(model.state_dict(), f'/models/{now}_{hyperparams["target"]}.pickle')
    
def train_loop(model, optimizer, scheduler, dataloader_train, dataloader_test, device, hyperparams):
    for epoch in range(hyperparams['num_epochs']):
        metric_logger = train(model, optimizer, scheduler, dataloader_train, device, epoch, hyperparams)
        print(metric_logger)
        
        if epoch % hyperparams['validation_interval'] == 0:
            validate(model, dataloader_test, device)
        
        if epoch % hyperparams['checkpoint_interval'] == 0:
            torch.save(model.state_dict(), f'/models/{hyperparams["time"]}_{hyperparams["target"]}.pickle')

def comet_train_loop(model, optimizer, scheduler, dataloader_train, dataloader_test, device, hyperparams):
    comet_ml.init()
    experiment = comet_ml.Experiment(api_key=hyperparams['comet_api_key'], project_name=hyperparams['comet_project_name'])
    experiment.log_parameters(hyperparams)
    
    for epoch in range(hyperparams['num_epochs']):
        with experiment.train():
            metric_logger = train(model, optimizer, scheduler, dataloader_train, device, epoch, hyperparams)
            print(metric_logger)
        
        with experiment.validate():
            validate(model, dataloader_test, device)
                
        if epoch % hyperparams['checkpoint_interval'] == 0:
            torch.save(model.state_dict(), f'/models/{hyperparams["time"]}_{hyperparams["target"]}.pickle')
        
    experiment.end()
    
def train(model, optimizer, scheduler, dataloader, device, epoch, hyperparams):
    result = train_one_epoch(model, optimizer, dataloader, device, epoch, hyperparams['print_freq'])
    scheduler.step()
    
    return result

def validate(model, dataloader, device):
    results = evaluate(model, dataloader, device=device)
    
    return results.evaluate()
