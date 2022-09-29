import sys
import os
import json
import random
import torch
import glob
import cv2

import numpy as np

from torch.utils.data import Dataset, DataLoader


class DataPartitioner:
    
    def __init__(self, parent_dir: str, random_seed: int):
        self.parent_dir = parent_dir
        self.random_seed = random_seed
        self.keys = os.listdir(parent_dir)
        
        # A dict of classes k containing child dicts with partitions
        self.data_dict = {k: self.split(self.read_dir(k)) for k in self.keys}
        
        # A dict of partitions containing child dicts with keys
        self.data_dict = {
            partition: {
                k: self.data_dict[k][partition]
                for k in self.data_dict
            }
            for partition in ['train', 'val', 'test']
        }
        
    def read_dir(self, subdir):
        """Returns a list of jpeg images filwes within the foder subdir into parent_dir"""
        return [
            os.path.basename(f)
            for f in glob.glob(os.path.join(self.parent_dir, subdir, '*.jpeg'))
        ]
        
    def split(
        self, lst,
        percents = [0.7, 0.2, 1]
    ):
        """
        """
        train_perc , vali_perc , test_perc = percents
        total = train_perc + vali_perc + test_perc
        part1 = int(len(lst)*train_perc/total)
        part2 = int(len(lst)*(train_perc+vali_perc)/total)
        
        random.seed(self.random_seed)
        random.shuffle(lst)
        
        partition_dict = {
            'train': lst[0:part1],
            'val': lst[part1:part2],
            'test': lst[part2:]
        }
        
        return partition_dict


#####################################################################
# DATASET
#####################################################################

class CustomDataset(Dataset):
    
    def __init__(
        self,
        parent_dir,
        data_dict,
        partition,
        transform = None,
        transform_workaround = None
    ):
        self.parent_dir = parent_dir
        self.data_dict = data_dict[partition]
        self.partition = partition
        self.transform = transform
        self.transform_workaround = transform_workaround
        self.keys = {k: enu for enu,k in enumerate(self.data_dict)}
        self.key_len = {k: len(self.data_dict[k]) for k in self.keys}
        
        total_len = 0
        for k in self.keys:
            total_len += self.key_len[k]
        
        self.total_len = total_len
        self.weights = [1-(self.key_len[k] / self.total_len) for k in self.keys]
        
    def __len__(self):
        return self.total_len
    
    def __getitem__( self, ii ):
        
        try:
        
            # ii is the global index considering files from all kinds
            # i is the index for the specific kind that it belongs to

            i = int(ii)

            for enu,kind in enumerate(self.keys):

                if i < self.key_len[kind]:

                    # current i,k and enu are the from the kind that belongs
                    kind_number = self.keys[kind]
                    basename = self.data_dict[kind][i]
                    break

                else:
                    i = i - self.key_len[kind]

                    # if newi < 0:  # skept group of indexes (kind) where it belongs.
                    #     # do not update i further because this one is beyond its kind
                    #     break
                    # else:
                    #     i = newi

            # i is already corrumpted at this point (we don't longer use it)

            img_filename = os.path.join(
                self.parent_dir,
                kind,
                basename
            )
            
            sample = {
                'inimg': np.transpose(
                    cv2.resize(cv2.imread(img_filename), (112, 112)),
                    [2,0,1]
                )
            }

            if self.transform:

                sample = {
                    'inimg': self.transform( sample['inimg'] ),
                }

            if self.transform_workaround:

                if 'Divide' in self.transform_workaround:

                    div = self.transform_workaround['Divide']

                    sample['inimg'] = sample['inimg']/div

                if 'Normalize' in self.transform_workaround:

                    mean_lst,std_lst = self.transform_workaround['Normalize']

                    sample = {
                        'inimg': np.stack( [(sample['inimg'][i,...]-mean_lst[i])/std_lst[i] for i in range(3)] , axis=0 ),
                    }

                if 'RandomHorizontalFlip' in self.transform_workaround:

                    if np.random.rand() > 0.5:

                        sample['inimg'] = sample['inimg'][...,::-1]

                if 'RandomVerticalFlip' in self.transform_workaround:

                    if np.random.rand() > 0.5:

                        sample['inimg'] = sample['inimg'][...,::-1,:]

                if 'RandomRotation' in self.transform_workaround:

                    if np.random.rand() > 0.5:

                        sample['inimg'] = np.transpose( sample['inimg'] , [0,2,1] )

            sample = {
                'inimg': torch.from_numpy(sample['inimg'].astype(np.float32)),
                'kind_number': kind_number,
                'kind': kind,
                'basename': basename
            }
            
        except:
            
            print('dataloader ERROR: ',i, self.key_len[kind],ii,self.partition, kind)
            print( [f for f in self.data_dict[kind] if not f.startswith(kind[0])] )
            print(self.data_dict[kind][i])
            print(basename)
            
        return sample


#####################################################################
# DATALOADER
#####################################################################

def get_dataloader(
    parent_dir='data',
    partition = 'train',
    batch_size=5,
    num_workers=4,
    random_seed=54
):
    data_partitioner = DataPartitioner(
        parent_dir=parent_dir, random_seed=random_seed)
    
    bool_flp_rot = (True if partition=='train' else False)
    
    dataset = CustomDataset(
        data_partitioner.parent_dir,
        data_partitioner.data_dict,
        partition,
        transform_workaround = {
            'Divide': 255,
            'RandomHorizontalFlip': bool_flp_rot,
            'RandomVerticalFlip': bool_flp_rot,
            'RandomRotation': bool_flp_rot
        }

    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    return dataloader
