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


def test_core(
    outpath,
    model,
    test_dataloader,
    weights_fn = './last.pt',
    device = 'cuda' if torch.cuda.is_available() else 'cpu',
    criterion = nn.MSELoss(),
    metrics_lst = [],
    metrics_func_lst = [],
):
    
    os.makedirs( os.path.dirname( weights_fn ) , exist_ok=True )
    
    model = model.to(device)
    
    mode_lst = ['test']
    
    metrics_lst = ['loss'] + metrics_lst
    metrics_func_lst = [criterion] + metrics_func_lst

    metrics_dict = {
        mode:{ met:[] for met in metrics_lst }
        for mode in mode_lst
    }
    
    for epoch in range(n_epochs):

        aux = model.eval()
        dataloader = test_dataloader
        metrics_batch = { met:[] for met in metrics_lst }
        
        for sample in dataloader:

            x = sample['inimg'].to( device )
            y = sample['kind_number'].to( device )

            pred = model.forward(x)
            loss = criterion( pred , y)

            cv2.imwrite(
                pred.detach().numpy(),
                os.path.join(outpath, sample['basename'])
            )
            
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

        for met in metrics_lst:
            
            metrics_dict[mode][met].append( np.mean(metrics_batch[met]) )

    results = {
        **{ 'test'+k:metrics_dict['test'][k] for k in metrics_dict['test']},
    }
    
    return results

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--weightsfn', type=str, default='last.pt', help='weights filename')
    parser.add_argument('--outpath', type=str, default='output_inference', help='weights filename')
    parser.add_argument('--model', type=str, default='resnet18', help='model name')
    parser.add_argument('--split_random_seed', type=int, default=54, help='split_random_seed')
    parser.add_argument('--batch_sz', type=int, default=1, help='batch_sz')
    parser.add_argument('--num_workers', type=int, default=1, help='num_workers')
    
    opt = parser.parse_args()

    # weights_fn = os.path.join(opt.experiment,
    #                           f'{opt.batch_sz}_{opt.n_epochs}_{opt.learning_rate}_{ahash}'
    #                          )

    print('Getting dataloaders...')
    
    test_dataloader = get_dataloader(
        partition = 'test',
        batch_size = opt.batch_sz,
        num_workers = opt.num_workers,
        random_seed = opt.split_random_seed
    )
    
    print('Building model...')
    
    model = build_model(
        opt.last_layer_num_neurons,
        name = opt.model,
        num_classes = len(train_dataloader.dataset.keys)
    )
    
    print('Start inference testing...')
    
 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    results = test_core(
        opt.outpath,
        model,
        test_dataloader,
        weights_fn = opt.weightsfn,
        device = 'cuda' if torch.cuda.is_available() else 'cpu',
        criterion = torch.nn.CrossEntropyLoss(),
        metrics_lst = [],
        metrics_func_lst = [],
    )
