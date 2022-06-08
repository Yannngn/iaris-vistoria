import comet_ml
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import ClassificationDataset
from utils import get_model_instance_classification

def main(data, experiment, hyperparams, device):
    model, transform_train, transform_val = get_model_instance_classification(hyperparams['num_classes'])
    model.to(device)

    # use our dataset and defined transformations
    dataset = ClassificationDataset(data, transform_train, hyperparams['target'])
    dataset_test = ClassificationDataset(data, transform_val, hyperparams['target'])

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-25])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-25:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=hyperparams['batch_size'], shuffle=True, num_workers=4)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=hyperparams['batch_size'], shuffle=False, num_workers=4)

    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
    
    confusion_matrix = experiment.create_confusion_matrix()

    dataiter = iter(data_loader_test)
    x_test, y_test = dataiter.next()
    y_pred = model(x_test.to(device)) 

    confusion_matrix.compute_matrix(y_test.data.numpy(), y_pred.data.cpu().numpy(), images=x_test.data.numpy(), image_channels='first')
    experiment.log_confusion_matrix(matrix=confusion_matrix, step=0, title=f"Confusion Matrix, Epoch 0", file_name=f"confusion-matrix-0.json", labels=uniques)
    callback = ConfusionMatrixCallbackReuseImages(model, experiment, x_test, y_test, confusion_matrix)
   
    step = 0
    for epoch in range(hyperparams['num_epochs']):
        print(f'Starting epoch {epoch}:')
        correct = 0
        total = 0
        epoch_loss = 0.
        with experiment.train():
            for i, (inputs, labels) in enumerate(data_loader, 0):
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(outputs.data, 1)
                batch_total = labels.size(0)
                total += batch_total

                batch_correct = (predicted == labels.data).sum()
                correct += batch_correct

                step += 1
                experiment.log_metric("batch_accuracy", batch_correct / batch_total, step=step)
                                
                # print statistics
                running_loss = loss.item()
                epoch_loss += running_loss
                
                print(f'mini batch {i} loss: {running_loss / (len(data_loader) * 5):.8f}')
        
                experiment.log_metric("batch_accuracy", correct / total, step=epoch)
                callback.on_epoch_end(epoch)
                print(f'epoch loss: {(epoch_loss / len(data_loader)):.8f}')

        with experiment.validate():
            y_pred = model(x_test.to(device))
            
            y_test = y_test.data
            _, predicted = torch.max(y_pred.data, 1)
            predicted = predicted.data.cpu().numpy()
            #val_acc = metrics.accuracy_score(y_test, y_pred)
            val_prec = metrics.precision_score(y_test, predicted, average='macro', zero_division=0)
            val_rec = metrics.recall_score(y_test, predicted, average='macro', zero_division=0)
            val_f1 = metrics.f1_score(y_test, predicted, average='macro', zero_division=0)

            #experiment.log_metric("accuracy", val_acc) 
            experiment.log_metric("precision", val_prec, step=step) 
            experiment.log_metric("recall", val_rec, step=step) 
            experiment.log_metric("f1", val_f1, step=step)     
    
    torch.save(model.state_dict(), PATH)

    print('Finished Training')
    
if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    
    main(hyperparams, device)