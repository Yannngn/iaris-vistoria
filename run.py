import os
import yaml

from datetime import datetime
from glob2 import glob

from dataset import get_images_and_labels_from_df
from detector import train_detector
from classifier import train_classifier
    
if __name__ == '__main__':
    abs_path = os.path.dirname(os.path.abspath(__file__))
    target = 'cor'
    model = 'class'
    project_name = f'model_{model}_{target}'
                                            
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    with open(os.path.join(abs_path, 'comet_config.yaml')) as f:
        comet = yaml.safe_load(f)
        comet['project_name'] = project_name
    
    if model == 'detect':
        with open(os.path.join(abs_path, 'detect_config.yaml')) as f:
            hyperparams = yaml.safe_load(f)
        hyperparams['target'] = target
        hyperparams['time'] = now
        
        images = glob(os.path.join(abs_path, 'data/images/')+'*.png')[:5]
        masks = []
        
        for k, v in comet.items():
            hyperparams[f'comet_{k}'] = v
            
        train_detector(images, masks, hyperparams)

    elif model == 'class':
        with open('classifier_config.yaml') as f:
            hyperparams = yaml.safe_load(f)
        hyperparams['target'] = target
        hyperparams['time'] = now
        for k, v in comet.items():
            hyperparams[f'comet_{k}'] = v
        
        images, labels = get_images_and_labels_from_df(os.path.join(abs_path, 'dataframe_carros_frente.csv'), 'resized', hyperparams['target'], hyperparams['labels'])
        #images = [i.replace('/content/drive/MyDrive/IARIS/computer-vision-team/benchmark/resize_segmentador_obrigatorios/resize_original', os.path.join(abs_path,'data/images')) for i in images]

        train_classifier(images, labels, hyperparams)
   