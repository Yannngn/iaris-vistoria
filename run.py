import os
import yaml

from datetime import datetime
from argparse import ArgumentParser

from dataset import get_images_and_labels_from_df

class Args:  
    @staticmethod
    def add_args(parent_parser: ArgumentParser) -> None:
        parser = ArgumentParser(parents=[parent_parser])

        parser.add_argument('--model', '-m', type=str, help="detection or classification")
        parser.add_argument('--name', '-n',  type=str, default='model')#, help="detection or classification")
        parser.add_argument('--target', '-t', type=str, default=None, help="farol, parabrisa, etc")
        parser.add_argument('--train_data', '-a', type=str, help="path to train data csv")
        parser.add_argument('--test_data', '-e', type=str, default=None, help="path to test data csv")
        
        return parser

def load_config(config_path):
    with open(config_path) as f:
        hyperparams = yaml.safe_load(f)
    
    if hparams.target: hyperparams['target'] = hparams.target
    if hparams.train_data: hyperparams['train_data'] = hparams.train_data
    if hparams.test_data: hyperparams['test_data'] = hparams.test_data
    if hparams.name: hyperparams['name'] = hparams.name

    hyperparams['project_name'] = f"{hyperparams['name']}_{hyperparams['model']}_{hyperparams['target']}"
    hyperparams['time'] = now
    
    return hyperparams

if __name__ == '__main__':
    parent_parser = ArgumentParser(add_help=False)

    parser = Args.add_args(parent_parser)
    hparams = parser.parse_args()
    
    colab = False
    abs_path = os.path.dirname(os.path.abspath(__file__))                                    
    now = datetime.now().strftime("%Y%m%d-%H%M%S")

    if hparams.model == 'detection':
        from detector import train_detector
        config_path = os.path.join(abs_path, 'detector_config.yaml')

        hyperparams = load_config(config_path)

        images, masks = get_images_and_labels_from_df(os.path.join(abs_path, hyperparams['train_data']), None, hyperparams)
        train_data = (images, masks)
        test_data = None
        
        if hyperparams['test_data']:
            images, masks = get_images_and_labels_from_df(os.path.join(abs_path, hyperparams['test_data']), None, hyperparams)
            test_data = (images, masks)
           
        train_detector(train_data, test_data, hyperparams)

    elif hparams.model == 'classification':
        from classifier import train_classifier
        config_path = os.path.join(abs_path, 'classifier_config.yaml')
        hyperparams = load_config(config_path)

        images, masks = get_images_and_labels_from_df(os.path.join(abs_path, hyperparams['train_data']), None, hyperparams)
        train_data = (images, masks)
        test_data = None
        
        if hyperparams['test_data']:
            images, masks = get_images_and_labels_from_df(os.path.join(abs_path, hyperparams['test_data']), None, hyperparams)
            test_data = (images, masks)
           
        train_classifier(train_data, test_data, hyperparams)

