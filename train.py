import sys
import os

os.environ['GIT_PYTHON_REFRESH'] = 'quiet'

import argparse
import json
import random
import torch
import glob
import mlflow
import torchmetrics

import numpy as np
import torch.nn.functional as F

from pthflops import count_ops
from torch import nn, optim

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score
)

from mlflow.tracking import MlflowClient

from datautils import get_dataloader
from makemodel import build_model


#####################################################################
# TRAIN LOOP
#####################################################################


def train_loop(
    model,
    train_dataloader,
    vali_dataloader,
    n_epochs = 300,
    weights_fn = './last.pt',
    optimizer = 'Here add something like > optim.Adam(model.parameters(), lr=0.0001)',
    device = 'cuda' if torch.cuda.is_available() else 'cpu',
    criterion = nn.MSELoss(),
    metrics_lst = [],
    metrics_func_lst = [],
    print_every_n_epochs = 10
):
    
    os.makedirs( os.path.dirname( weights_fn ) , exist_ok=True )
    
    model = model.to(device)
    
    mode_lst = ['train','vali']
    
    metrics_lst = ['loss'] + metrics_lst
    metrics_func_lst = [criterion] + metrics_func_lst

    metrics_dict = {
        mode:{ met:[] for met in metrics_lst }
        for mode in mode_lst
    }
    
    for epoch in range(n_epochs):

        for mode in mode_lst:

            if mode=='train':
                aux = model.train()
                dataloader = train_dataloader
            else:
                aux = model.eval()
                dataloader = vali_dataloader

            metrics_batch = { met:[] for met in metrics_lst }
            
            for sample in dataloader:

                x = sample['inimg'].to( device )
                y = sample['kind_number'].to( device )

                pred = model.forward(x)
                loss = criterion( pred , y)
                
                if mode=='train':
                    model.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                for f,met in zip( metrics_func_lst, metrics_lst ):
                    
                    if met=='loss':
                        metrics_batch[met].append( loss.item() )
                    else:
                        
                        one_hot_y = F.one_hot(y,num_classes=12)
                        
                        score = f(one_hot_y,pred)
                        
                        if score.isnan():
                            metrics_batch[met].append(0)
                        else:
                            metrics_batch[met].append( score )
                            
                        # metrics_batch[met].append( f(
                        #     one_hot_y, pred
                        # ).item() )
                        
#                         prednp = pred.cpu().detach().numpy()
#                         ynp = y.cpu().detach().numpy()
                        
#                         if len(ynp.shape) == 1:
#                             ynp = np.expand_dims(ynp, axis=0)
                        
#                         num_classes = len(dataloader.dataset.keys)
#                         ynp = np.eye(num_classes)[ynp.reshape(-1)]
                        
#                         prednp = np.transpose(prednp)
#                         ynp = np.transpose(ynp).astype('uint8')
                        
#                         roc = f(
#                             ynp,
#                             prednp,
#                             multi_class='ovr',
#                             labels=[e for e in dataloader.dataset.keys]
#                         )
                        
#                         metrics_batch[met].append( f(
#                             prednp,
#                             ynp,
#                             multi_class='ovr',
#                             labels=[e for e in dataloader.dataset.keys]
#                         ).item() )

            for met in metrics_lst:
                
                metrics_dict[mode][met].append( np.mean(metrics_batch[met]) )
                
        # Save weigths
        s_dict = model.state_dict()
        torch.save( s_dict , weights_fn )
        mlflow.pytorch.log_model(model, os.path.basename(weights_fn))
        
        if print_every_n_epochs:
            
            if epoch%print_every_n_epochs==0:
                
                print('*********************')
                print(f'epoch\t\t{epoch}')
                
                for mode in mode_lst:
                    
                    for met in metrics_lst:
                        
                        print(f'{mode}_{met}\t\t{metrics_dict[mode][met][-1]}')
            
        for mode in mode_lst:
                
            for met in ['loss']+metrics_lst:
                    
                curr_epoch_met = metrics_dict[mode][met][-1]
                # print(f"{mode}{met}", curr_epoch_met, epoch)
                # writer.add_scalar(f"{mode}/{met}", curr_epoch_met, epoch)
                mlflow.log_metric(f"{mode}M{met}", curr_epoch_met, epoch)
    
    results = {
        **{ 'train'+k:metrics_dict['train'][k] for k in metrics_dict['train']},
        **{ 'vali'+k:metrics_dict['vali'][k] for k in metrics_dict['vali']}
    }
    
    return results

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--split_random_seed', type=int, default=54, help='split_random_seed')
    parser.add_argument('--batch_sz', type=int, default=5, help='batch_sz')
    parser.add_argument('--n_epochs', type=int, default=10, help='n_epochs')
    parser.add_argument('--num_workers', type=int, default=1, help='num_workers')
    parser.add_argument('--last_layer_num_neurons', type=str, default='64',
        help='in case you want to set the number of layers for the first flat layer, specified as a comma separated string such as "64,128,256"')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning_rate')
    parser.add_argument('--experiment', type=str, default='hny', help='experiment_name for mlflow')
    parser.add_argument('--model', type=str, default='resnet18', help='model name')
    
    opt = parser.parse_args()
    
    os.environ['GIT_PYTHON_REFRESH'] = 'quiet'
    # os.environ['MLFLOW_TRACKING_URI'] = 'https://mlflow-heroku-honey.herokuapp.com/'
    os.environ['MLFLOW_TRACKING_USERNAME'] = 'honey'
    os.environ['MLFLOW_TRACKING_PASSWORD'] = 'honey' # This should not be here
    os.environ["MLFLOW_EXPERIMENT_NAME"] = opt.experiment

    #mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

    experiment_name = opt.experiment
    ahash = random.randint(0,999999999)
    weights_fn = os.path.join(
        opt.experiment,
        f'{opt.model}_{opt.batch_sz}_{ahash}.pt'
        )

    print('Getting dataloaders...')
    
    train_dataloader, val_dataloader =[
        get_dataloader(
            partition = partition,
            batch_size = opt.batch_sz,
            num_workers = opt.num_workers,
            random_seed = opt.split_random_seed
        ) for partition in ['train', 'val']
    ]
    
    print('Building model...')
    
    model = build_model(
        opt.last_layer_num_neurons,
        name = opt.model,
        num_classes = len(train_dataloader.dataset.keys)
    )
    
    print('Start training...')
    
    client = MlflowClient()
    
    try:
        experiment_id = client.create_experiment(experiment_name)
    except:
        # already exists
        current_experiment=dict(mlflow.get_experiment_by_name(experiment_name))
        experiment_id=current_experiment['experiment_id']
    
    with mlflow.start_run(experiment_id = experiment_id):
        
        mlflow.log_param('batch',opt.batch_sz)
        mlflow.log_param('epoch',opt.n_epochs)
        mlflow.log_param('lr',opt.learning_rate)
        mlflow.log_param('SplitRndSeed',opt.split_random_seed)
        mlflow.log_param('NeuronsLast', opt.last_layer_num_neurons)
        mlflow.log_param('model', opt.model)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Count the number of FLOPs
        model.to(device)
        flops = count_ops(model, torch.rand(1,3,112,112).to(device))[0]
        mlflow.log_param('GFLOPS', flops/10e9 )
        
        results = train_loop(
            model,
            train_dataloader,
            val_dataloader,
            n_epochs             = opt.n_epochs,
            weights_fn           = weights_fn,
            optimizer            = optim.Adam(model.parameters(), lr=opt.learning_rate),
            device               = device,
            criterion            = torch.nn.CrossEntropyLoss(
                weight=torch.Tensor(train_dataloader.dataset.weights).to(device)),
            metrics_lst          = [
                #'ROC',
                #'AP',
            ],
            metrics_func_lst     = [
                #roc_auc_score
                #torchmetrics.AveragePrecision(num_classes=len(train_dataloader.dataset.keys))
            ],
            print_every_n_epochs = 1
        )
