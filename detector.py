import comet_ml
import numpy as np
import torch

from datetime import datetime
from torchvision import ops
from tqdm import tqdm

from scripts.engine import train_one_epoch, evaluate

from dataset import get_data
from utils import get_model_instance_detection, get_optimizer, get_scheduler, mask_iou

def train_detector(train_data, test_data, hyperparams, comet=True):
    device = torch.device('cpu')
    if hyperparams['device'] == 'gpu':
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model, train_transform, test_transform = get_model_instance_detection(hyperparams)
    model.to(device)
    
    dataloader_train, dataloader_test = get_data(train_data, test_data, train_transform, test_transform, hyperparams)
    
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
    
    torch.save(model.state_dict(), f'models/{now}_{hyperparams["target"]}.pickle')
    
def train_loop(model, optimizer, scheduler, dataloader_train, dataloader_test, device, hyperparams):
    for epoch in range(hyperparams['num_epochs']):
        metric_logger = train(model, optimizer, scheduler, dataloader_train, device, epoch, hyperparams)
        print(metric_logger)
        
        if epoch % hyperparams['validation_interval'] == 0 and epoch != 0:
            evaluate(model, dataloader_test, device=device)
            validate(model, dataloader_test, device)
        
        if epoch % hyperparams['checkpoint_interval'] == 0 or epoch == hyperparams['num_epochs']:
            torch.save(model.state_dict(), f'models/{hyperparams["time"]}_{hyperparams["project_name"]}.pickle')

        evaluate(model, dataloader_test, device=device)

def comet_train_loop(model, optimizer, scheduler, dataloader_train, dataloader_test, device, hyperparams):
    comet_ml.init()
    experiment = comet_ml.Experiment(api_key=hyperparams['api_key'], project_name=hyperparams['project_name'])
    experiment.log_parameters(hyperparams)
    
    for epoch in range(hyperparams['num_epochs']):
        experiment.set_epoch(epoch)
        experiment.set_step(epoch)
        print(f'Starting training epoch {epoch}:') 
        
        with experiment.train():
            metric_logger = train(model, optimizer, scheduler, dataloader_train, device, epoch, hyperparams)
            print(metric_logger)
        
        with experiment.validate():
            
            metrics = {}
            metrics['num_objs'], metrics['num_preds'], metrics['boxes_iou'], metrics['masks_iou'] = validate(model, dataloader_test, device)

            experiment.log_metrics(metrics)
              
        if epoch % hyperparams['checkpoint_interval'] == 0 or epoch == hyperparams['num_epochs']:
            torch.save(model.state_dict(), f'models/{hyperparams["time"]}_{hyperparams["project_name"]}.pickle')
        
        evaluate(model, dataloader_test, device=device)
        
    experiment.end()
    
def train(model, optimizer, scheduler, dataloader, device, epoch, hyperparams):
    result = train_one_epoch(model, optimizer, dataloader, device, epoch, hyperparams['print_freq'])
    scheduler.step()
    
    return result

def validate(model, dataloader, device):
    # calc box IoU
    # calc mask IoU
    # multiply individual IoU by the 1 / quantity of objs
    # if there are more predictions than objects, multiply next scores by -1 
    num_objs, num_preds = 0, 0
    boxes_iou, masks_iou = 0., 0.
    loader = tqdm(dataloader)
    with torch.no_grad():
        model.eval()
        for _, (inputs, targets) in enumerate(loader):
            inputs, boxes, masks = inputs.to(device), targets['boxes'].to(device), targets['masks'].to(device)
            outputs = model(inputs)

            num_objs += len(masks)        

            count = 0
            for i, score in enumerate(outputs['scores']):
                if score > .5:
                    mult = 1 if count <= num_objs else -1
                    
                    box_iou += ops.box_iou(outputs['boxes'][i], boxes[i]) * mult / num_objs

                    mask_iou += mask_iou(outputs['masks'][i] > .5, masks[i]) * mult / num_objs
                    
                    count += 1

            num_preds += count

    print(f'''Segmented Objects: {num_objs}\n 
              Model predicted: {num_preds}\n
              Mean BBox IoU: {boxes_iou}\n
              Mean Mask IoU: {masks_iou}
              ''')

    return num_objs, num_preds, boxes_iou, masks_iou




