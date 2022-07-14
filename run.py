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
    
    run_type = input('detect or classify')

    if run_type == 'detection':
        with open(os.path.join(abs_path, 'detector_config.yaml')) as f:
            hyperparams = yaml.safe_load(f)
            
        hyperparams['project_name'] = f"model_{hyperparams['model']}_{hyperparams['target']}"
        hyperparams['time'] = now
    
        #images, masks = get_images_and_labels_from_df(os.path.join(abs_path, comet['data']), os.path.join(abs_path, r'data\obrigatorios'), hyperparams)
        
        images, masks = get_images_and_labels_from_df(os.path.join(abs_path, hyperparams['data']), None, hyperparams)
            
        train_detector(images, masks, hyperparams)

    elif run_type == 'classification':
        with open('classifier_config.yaml') as f:
            hyperparams = yaml.safe_load(f)
            
        hyperparams['project_name'] = f"model_{hyperparams['model']}_{hyperparams['target']}"
        hyperparams['time'] = now

        
        #images, labels = get_images_and_labels_from_df(os.path.join(abs_path, comet['data']), os.path.join(abs_path, r'data\super'), hyperparams)
        images, labels = get_images_and_labels_from_df(os.path.join(abs_path, hyperparams['data']), None, hyperparams)
        
        train_classifier(images, labels, hyperparams)
   