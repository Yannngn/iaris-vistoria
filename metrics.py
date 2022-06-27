import torch

def evaluate_classification(prob: torch.tensor, label: torch.tensor, global_metrics, label_metrics, hyperparams) -> dict:
    num_classes = hyperparams['num_classes']
    uniques = hyperparams['labels']
    dict_eval = {}

    for (metric, f) in global_metrics.items():
        result = f(prob, label).cpu().numpy()
        if metric == 'stats':
            dict_eval[f'{metric}_tp'] = 0. if result[0] != result[0] else result[0]
            dict_eval[f'{metric}_fp'] = 0. if result[1] != result[1] else result[1]
            dict_eval[f'{metric}_tn'] = 0. if result[2] != result[2] else result[2]
            dict_eval[f'{metric}_fn'] = 0. if result[3] != result[3] else result[3]
            dict_eval[f'{metric}_sup'] = 0. if result[4] != result[4] else result[4]
        else:
            dict_eval[f'{metric}'] = 0. if result != result else result
        
    for (metric, f) in label_metrics.items():
        result = f(prob, label).cpu().numpy()
        for i in range(num_classes):
            dict_eval[f'{uniques[i]}_{metric}'] = 0. if result[i] != result[i] else result[i]
        
    return dict_eval