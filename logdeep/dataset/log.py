#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler


class log_dataset(Dataset):
    def __init__(self, logs, labels, seq=True, quan=False, sem=False):
        self.seq = seq
        self.quan = quan
        self.sem = sem
        if self.seq:
            self.Sequentials = logs['Sequentials']
        if self.quan:
            self.Quantitatives = logs['Quantitatives']
        if self.sem:
            self.Semantics = logs['Semantics']
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        log = dict()
        log['Sequentials'] = []
        if self.seq:
            logdata = torch.tensor(self.Sequentials[idx],
                                              dtype=torch.float)
            # log['Sequentials'] = torch.tensor(self.Sequentials[idx],
            #                                   dtype=torch.float)
            # log1 = self.rand_mask(logdata)
            # log2 = self.rand_mask(logdata)
        if self.quan:
            log['Quantitatives'] = torch.tensor(self.Quantitatives[idx],
                                                dtype=torch.float)
        if self.sem:
            log['Semantics'] = torch.tensor(self.Semantics[idx],
                                            dtype=torch.float)
        return logdata, self.labels[idx]

    def rand_mask(self, log):
        n = len(log)
        p = int(n * 0.4)
        masknum = np.random.choice(n, size=p, replace=False)
        mask = torch.ones(n)
        mask[masknum] = 0
        
        masked_data = log.T * mask
        return masked_data.T


if __name__ == '__main__':
    data_dir = '../../data/hdfs/hdfs_train'
    window_size = 10
    train_logs = prepare_log(data_dir=data_dir,
                             datatype='train',
                             window_size=window_size)
    train_dataset = log_dataset(log=train_logs, seq=True, quan=True)
    print(train_dataset[0])
    print(train_dataset[100])
