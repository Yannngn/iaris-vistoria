import os
import yaml

from datetime import datetime
from argparse import ArgumentParser

from dataset import get_images_and_labels_from_df
from detector import train_detector
from classifier import train_classifier

class Args:  
    @staticmethod
    def add_args(parent_parser: ArgumentParser) -> None:
        parser = ArgumentParser(parents=[parent_parser])

        parser.add_argument('--model', '-m', type=str, help="detection or classification")
        parser.add_argument('--target', '-t', type=str, help="farol, parabrisa, etc")
        parser.add_argument('--in_path', '-i', type=str, help="path to csv")
        
        return parser

if __name__ == '__main__':
    parent_parser = ArgumentParser(add_help=False)

    parser = Args.add_args(parent_parser)
    hparams = parser.parse_args()
    
    colab = False
    abs_path = os.path.dirname(os.path.abspath(__file__))                                    
    now = datetime.now().strftime("%Y%m%d-%H%M%S")

    if hparams.model == 'detection':
        with open(os.path.join(abs_path, 'detector_config.yaml')) as f:
            hyperparams = yaml.safe_load(f)
            hyperparams['target'] = hparams.target
            hyperparams['data'] = hparams.in_path
        hyperparams['project_name'] = f"model_{hyperparams['model']}_{hyperparams['target']}"
        hyperparams['time'] = now


        images, masks = get_images_and_labels_from_df(os.path.join(abs_path, hyperparams['train_data']), None, hyperparams)
        train_data = (images, masks)
        test_data = None
        
        if hyperparams['test_data']:
            images, masks = get_images_and_labels_from_df(os.path.join(abs_path, hyperparams['test_data']), None, hyperparams)
            test_data = (images, masks)
           
        train_detector(train_data, test_data, hyperparams)

    elif hparams.model == 'classification':
        with open('classifier_config.yaml') as f:
            hyperparams = yaml.safe_load(f)
            
        hyperparams['project_name'] = f"model_{hyperparams['model']}_{hyperparams['target']}"
        hyperparams['time'] = now

        images, masks = get_images_and_labels_from_df(os.path.join(abs_path, hyperparams['train_data']), None, hyperparams)
        train_data = (images, masks)
        test_data = None
        
        if hyperparams['test_data']:
            images, masks = get_images_and_labels_from_df(os.path.join(abs_path, hyperparams['test_data']), None, hyperparams)
            test_data = (images, masks)
           
        train_classifier(train_data, test_data, hyperparams)

