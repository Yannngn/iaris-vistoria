import os
import yaml

from datetime import datetime

from dataset import get_images_and_labels_from_df
from detector import train_detector
from classifier import train_classifier

if __name__ == '__main__':
    colab = False
    abs_path = os.path.dirname(os.path.abspath(__file__))                                    
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    with open(os.path.join(abs_path, 'run_config.yaml')) as f:
        comet = yaml.safe_load(f)
    
    target = comet['target']
    model = comet['model']

    comet['project_name'] = f"model_{comet['model']}_{comet['target']}"

    if comet['model'] == 'detection':
        with open(os.path.join(abs_path, 'detector_config.yaml')) as f:
            hyperparams = yaml.safe_load(f)
        hyperparams['target'] = target
        hyperparams['time'] = now
    
        images, masks = get_images_and_labels_from_df(os.path.join(abs_path, comet['data']), os.path.join(abs_path, r'data\obrigatorios'), hyperparams)
        
        #images = [i.replace('/content/drive/MyDrive/IARIS/computer-vision-team/benchmark/resize_segmentador_obrigatorios/resize_original', os.path.join(abs_path,'obrigatorios/data/images')) for i in images]
        #masks = [m.replace('/content/drive/MyDrive/IARIS/computer-vision-team/benchmark/resize_segmentador_obrigatorios/combined_masks', os.path.join(abs_path,'obrigatorios/data/masks')) for m in masks]
        
        for k, v in comet.items():
            hyperparams[f'comet_{k}'] = v
            
        train_detector(images, masks, hyperparams)

    elif model == 'classification':
        with open('classifier_config.yaml') as f:
            hyperparams = yaml.safe_load(f)
        hyperparams['target'] = target
        hyperparams['time'] = now
        
        for k, v in comet.items():
            hyperparams[f'comet_{k}'] = v
        
        images, labels = get_images_and_labels_from_df(os.path.join(abs_path, comet['data']), os.path.join(abs_path, r'data\super'), hyperparams)
        #images = [i.replace('/content/drive/MyDrive/IARIS/computer-vision-team/benchmark/resize_segmentador_obrigatorios/resize_original', os.path.join(abs_path,'super/data/images')) for i in images]
        
        train_classifier(images, labels, hyperparams)
   