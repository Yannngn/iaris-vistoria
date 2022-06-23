import yaml

from detector import train_detector
from classifier import train_classifier
    
if __name__ == '__main__':
    images = []
    masks = []
    
    target = 'farol'
    model = 'detect'
    project_name = f'model_{model}_{target}'
    
    with open('comet_config.yaml') as f:
        comet = yaml.safe_load(f, Loader=yaml.FullLoader)
        comet['project_name'] = project_name
    
    if model == 'detect':
        with open('detect_config.yaml') as f:
            hyperparams = yaml.safe_load(f, Loader=yaml.FullLoader)
            hyperparams['target'] = target
            train_detector(images, masks, hyperparams)

    elif model == 'class':
        with open('classifier_config.yaml') as f:
            hyperparams = yaml.safe_load(f, Loader=yaml.FullLoader)
            hyperparams['target'] = target
            train_classifier(images, masks, hyperparams)
   