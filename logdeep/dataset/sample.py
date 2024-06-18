import json
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm


def read_json(filename):
    with open(filename, 'r') as load_f:
        file_dict = json.load(load_f)
    return file_dict


def trp(l, n):
    """ Truncate or pad a list """
    r = l[:n]
    if len(r) < n:
        r.extend(list([0]) * (n - len(r)))
    return r


def down_sample(logs, labels, sample_ratio):
    print('sampling...')
    total_num = len(labels)
    all_index = list(range(total_num))
    sample_logs = {}
    for key in logs.keys():
        sample_logs[key] = []
    sample_labels = []
    sample_num = int(total_num * sample_ratio)

    for i in tqdm(range(sample_num)):
        random_index = int(np.random.uniform(0, len(all_index)))
        for key in logs.keys():
            sample_logs[key].append(logs[key][random_index])
        sample_labels.append(labels[random_index])
        del all_index[random_index]
    return sample_logs, sample_labels

def sample_num(data_dir, window_size, num, result_logs, labels, event2semantic_vec, num_sessions=0):
    with open(data_dir, 'r') as f:
        for line in f.readlines():
            if num_sessions >= num+10:
                break
            num_sessions += 1
            if num_sessions < 10 : continue
            line = tuple(map(lambda n: n - 1, map(int, line.strip().split())))
            for i in range(len(line) - window_size):
                Sequential_pattern = list(line[i:i + window_size])
                Quantitative_pattern = [0] * 28
                log_counter = Counter(Sequential_pattern)

                for key in log_counter:
                    Quantitative_pattern[key] = log_counter[key]
                Semantic_pattern = []
                for event in Sequential_pattern:
                    if event == 0:
                        Semantic_pattern.append([-1] * 300)
                    else:
                        Semantic_pattern.append(event2semantic_vec[str(event -
                                                                       1)])
                Sequential_pattern = np.array(Sequential_pattern)[:,
                                                                  np.newaxis]
                Quantitative_pattern = np.array(
                    Quantitative_pattern)[:, np.newaxis]
                
                result_logs['Sequentials'].append(Sequential_pattern)
                result_logs['Quantitatives'].append(Quantitative_pattern)
                result_logs['Semantics'].append(Semantic_pattern)
                labels.append(line[i + window_size])
    
    return result_logs, labels


def sliding_window(data_dir, datatype, window_size, clientnum, sample_ratio=1):
    '''
    dataset structure
        result_logs(dict):
            result_logs['feature0'] = list()
            result_logs['feature1'] = list()
            ...
        labels(list)
    '''
    event2semantic_vec = read_json(data_dir + 'hdfs/event2semantic_vec.json')
    num_sessions = 0
    Sequential_pattern = []
    
    
    if datatype == 'train':
        data_dir += 'hdfs/hdfs_train'
        labels = [[] for i in range(clientnum)]
        resultlist = []
        for i in range(clientnum):
            result_logs = {}
            result_logs['Sequentials'] = []
            result_logs['Quantitatives'] = []
            result_logs['Semantics'] = []
            resultlist.append(result_logs)
    if datatype == 'val':
        labels = []
        result_logs = {}
        result_logs['Sequentials'] = []
        result_logs['Quantitatives'] = []
        result_logs['Semantics'] = []
        data_dir += 'hdfs/hdfs_test_normal'
    if datatype == 'share':
        labels = []
        result_logs = {}
        result_logs['Sequentials'] = []
        result_logs['Quantitatives'] = []
        result_logs['Semantics'] = []
        data_dir_norl = data_dir + 'hdfs/hdfs_test_normal'
        data_dir_ab = data_dir + 'hdfs/hdfs_test_abnormal'

    
    if datatype == 'val' or datatype == 'train':
        with open(data_dir, 'r') as f:
            for line in f.readlines():
                clientid = num_sessions % clientnum
                num_sessions += 1
                line = tuple(map(lambda n: n - 1, map(int, line.strip().split())))
                for i in range(len(line) - window_size):
                    Sequential_pattern = list(line[i:i + window_size])
                    Quantitative_pattern = [0] * 28
                    log_counter = Counter(Sequential_pattern)
    
                    for key in log_counter:
                        Quantitative_pattern[key] = log_counter[key]
                    Semantic_pattern = []
                    for event in Sequential_pattern:
                        if event == 0:
                            Semantic_pattern.append([-1] * 300)
                        else:
                            Semantic_pattern.append(event2semantic_vec[str(event -
                                                                           1)])
                    Sequential_pattern = np.array(Sequential_pattern)[:,
                                                                      np.newaxis]
                    Quantitative_pattern = np.array(
                        Quantitative_pattern)[:, np.newaxis]
                    if datatype == 'train':
                        resultlist[clientid]['Sequentials'].append(Sequential_pattern)
                        resultlist[clientid]['Quantitatives'].append(Quantitative_pattern)
                        resultlist[clientid]['Semantics'].append(Semantic_pattern)
                        labels[clientid].append(line[i + window_size])
                    if datatype == 'val':
                        result_logs['Sequentials'].append(Sequential_pattern)
                        result_logs['Quantitatives'].append(Quantitative_pattern)
                        result_logs['Semantics'].append(Semantic_pattern)
                        labels.append(line[i + window_size])
    if datatype == 'share':
        result_logs, labels = sample_num(data_dir_norl, window_size, 10, result_logs, labels, event2semantic_vec, 0)
        result_logs, labels = sample_num(data_dir_ab, window_size, 10, result_logs, labels, event2semantic_vec, 0)
                

    if sample_ratio != 1:
        if datatype == 'train':
            for clientid in range(clientnum):
                resultlist[clientid], labels[clientid] = down_sample(resultlist[clientid], labels[clientid], sample_ratio)
        if datatype == 'val':
            result_logs, labels = down_sample(result_logs, labels, sample_ratio)
        if datatype == 'share':
            result_logs, labels = down_sample(result_logs, labels, sample_ratio)
    if datatype == 'train':
        print(f'File {data_dir}, number of sessions {num_sessions}')
        for clientid in range(clientnum):
            print(f'number of client {clientid} seqs {len(labels[clientid])}')
        return resultlist, labels
    if datatype == 'val':
        print('File {}, number of seqs {}'.format(data_dir,len(result_logs['Sequentials'])))
        return result_logs, labels
    if datatype == 'share':
        print('Shared data number of seqs {}'.format(len(labels)))
        return result_logs, labels

    


def session_window(data_dir, datatype, sample_ratio=1):
    event2semantic_vec = read_json(data_dir + 'hdfs/event2semantic_vec.json')
    result_logs = {}
    result_logs['Sequentials'] = []
    result_logs['Quantitatives'] = []
    result_logs['Semantics'] = []
    labels = []

    if datatype == 'train':
        data_dir += 'hdfs/robust_log_train.csv'
    elif datatype == 'val':
        data_dir += 'hdfs/robust_log_valid.csv'
    elif datatype == 'test':
        data_dir += 'hdfs/robust_log_test.csv'

    train_df = pd.read_csv(data_dir)
    for i in tqdm(range(len(train_df))):
        ori_seq = [
            int(eventid) for eventid in train_df["Sequence"][i].split(' ')
        ]
        Sequential_pattern = trp(ori_seq, 50)
        Semantic_pattern = []
        for event in Sequential_pattern:
            if event == 0:
                Semantic_pattern.append([-1] * 300)
            else:
                Semantic_pattern.append(event2semantic_vec[str(event - 1)])
        Quantitative_pattern = [0] * 29
        log_counter = Counter(Sequential_pattern)

        for key in log_counter:
            Quantitative_pattern[key] = log_counter[key]

        Sequential_pattern = np.array(Sequential_pattern)[:, np.newaxis]
        Quantitative_pattern = np.array(Quantitative_pattern)[:, np.newaxis]
        result_logs['Sequentials'].append(Sequential_pattern)
        result_logs['Quantitatives'].append(Quantitative_pattern)
        result_logs['Semantics'].append(Semantic_pattern)
        labels.append(int(train_df["label"][i]))

    if sample_ratio != 1:
        result_logs, labels = down_sample(result_logs, labels, sample_ratio)

    # result_logs, labels = up_sample(result_logs, labels)

    print('Number of sessions({}): {}'.format(data_dir,
                                              len(result_logs['Semantics'])))
    return result_logs, labels
