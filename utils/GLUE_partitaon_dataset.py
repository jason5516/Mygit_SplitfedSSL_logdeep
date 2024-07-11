from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
import pickle

# 从文件加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained("./llm_models/roberta_base")
# tokenizer = AutoTokenizer.from_pretrained("roberta-base")

# 定義 tokenization 函數
def mrpc_tokenize_function(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=128)
def sst2_tokenize_function(examples):
    return tokenizer(examples["sentence"], truncation=True, max_length=128)
def rte_tokenize_function(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=128)
def mnli_tokenize_function(examples):
    return tokenizer(examples["premise"], examples["hypothesis"], truncation=True, max_length=128)
def qnli_tokenize_function(examples):
    return tokenizer(examples["question"], examples["sentence"], truncation=True, max_length=128)
def wnli_tokenize_function(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=128)
def qqp_tokenize_function(examples):
    return tokenizer(examples["question1"], examples["question2"], truncation=True, max_length=128)

def load_partition_glue_data(data="mrpc", client_num=10, alpha=1.0):
    # 加載 IMDB 資料集
    # "wnli" 有問題的資料集
    # glue dataset ["mrpc", "sst2", "rte", "mnli", "qnli", "qqp"]
    
    # 读取 data.pkl 文件
    data_path = "./glue_data/" + data + "_data.pkl"
    partition_path = "./glue_data/" + data + "_partition.pkl"
    attr = "clients=" + str(client_num) + "_alpha=" + str(alpha)
    
    with open(data_path, 'rb') as file:
        raw_data = pickle.load(file)
    with open(partition_path, 'rb') as file:
        partition_data = pickle.load(file)
    dataset = load_dataset("glue", data)    

    npdata_train = np.array(dataset["train"][:]["sentence1"])
    npdata_valid = np.array(dataset["validation"][:]["sentence1"])
    npdata_test = np.array(dataset["test"][:]["sentence1"])
    
    # 將資料根據pkl檔案重新排列
    raw_train_idx = []
    for i in raw_data["train"]:
        target_a = i.text_a
        raw_train_idx.append(np.where(npdata_train == target_a)[0].item())
    raw_valid_idx = []
    for i in raw_data["valid"]:
        target_a = i.text_a
        raw_valid_idx.append(np.where(npdata_train == target_a)[0].item())
    raw_test_idx = []
    for i in raw_data["test"]:
        target_a = i.text_a
        raw_test_idx.append(np.where(npdata_valid == target_a)[0].item())
    
    
    # 對資料集進行 tokenization
    if data == "mrpc":
        tokenized_datasets = dataset.map(mrpc_tokenize_function, batched=True)
        # 移除無用的列，只保留 input_ids 和 labels
        tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        tokenized_datasets.set_format("torch")
    elif data == "sst2":
        tokenized_datasets = dataset.map(sst2_tokenize_function, batched=True)
        # 移除無用的列，只保留 input_ids 和 labels
        tokenized_datasets = tokenized_datasets.remove_columns(["sentence", "idx"])
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        tokenized_datasets.set_format("torch")
    elif data == "rte":
        tokenized_datasets = dataset.map(rte_tokenize_function, batched=True)
        # 移除無用的列，只保留 input_ids 和 labels
        tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        tokenized_datasets.set_format("torch")
    elif data == "mnli":
        tokenized_datasets = dataset.map(mnli_tokenize_function, batched=True)
        # 移除無用的列，只保留 input_ids 和 labels
        tokenized_datasets = tokenized_datasets.remove_columns(["premise", "hypothesis", "idx"])
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        tokenized_datasets.set_format("torch")
    elif data == "qnli":
        tokenized_datasets = dataset.map(qnli_tokenize_function, batched=True)
        # 移除無用的列，只保留 input_ids 和 labels
        tokenized_datasets = tokenized_datasets.remove_columns(["question", "sentence", "idx"])
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        tokenized_datasets.set_format("torch")
    elif data == "wnli":
        tokenized_datasets = dataset.map(wnli_tokenize_function, batched=True)
        # 移除無用的列，只保留 input_ids 和 labels
        tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        tokenized_datasets.set_format("torch")
    elif data == "qqp":
        tokenized_datasets = dataset.map(qqp_tokenize_function, batched=True)
        # 移除無用的列，只保留 input_ids 和 labels
        tokenized_datasets = tokenized_datasets.remove_columns(["question1", "question2", "idx"])
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        tokenized_datasets.set_format("torch")
    else:
        tokenized_datasets = dataset.map(mrpc_tokenize_function, batched=True)
        # 移除無用的列，只保留 input_ids 和 labels
        tokenized_datasets = tokenized_datasets.remove_columns(["text", "idx"])
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        tokenized_datasets.set_format("torch")

    
    
    # 將資料根據劃分分給客戶
    client_datasets = [{} for i in range(client_num)]
    for cid in range(client_num):
        train_temp = partition_data[attr]["train"][cid]
        valid_temp = partition_data[attr]["valid"][cid]
        test_temp = partition_data[attr]["test"][cid]
        train_idx = [raw_train_idx[i] for i in train_temp]
        valid_idx = [raw_valid_idx[i] for i in valid_temp]
        test_idx = [raw_test_idx[i] for i in test_temp]
        client_datasets[cid]["train"] = tokenized_datasets["train"].select(train_idx)
        client_datasets[cid]["valid"] = tokenized_datasets["train"].select(valid_idx)
        client_datasets[cid]["test"] = tokenized_datasets["validation"].select(test_idx)

    # 列印客戶端labels不平衡程度
    for i in range(10):
        train_temp = client_datasets[i]["train"]["labels"]
        valid_temp = client_datasets[i]["valid"]["labels"]
        test_temp = client_datasets[i]["test"]["labels"]
        train_label_1 = len((list(filter(lambda x: x == 1, train_temp))))
        train_label_0 = len((list(filter(lambda x: x == 0, train_temp))))
        valid_label_1 = len((list(filter(lambda x: x == 1, valid_temp))))
        valid_label_0 = len((list(filter(lambda x: x == 0, valid_temp))))
        test_label_1 = len((list(filter(lambda x: x == 1, test_temp))))
        test_label_0 = len((list(filter(lambda x: x == 0, test_temp))))
        print(f"client {i} \n train({train_label_1+train_label_0}) | 1 : {train_label_1}, 0 : {train_label_0} |, " +
              f"valid({valid_label_1+valid_label_0}) | 1 : {valid_label_1}, 0 : {valid_label_0} |, " +
              f"test({test_label_1+test_label_0}) | 1 : {test_label_1}, 0 : {test_label_0} |"
             )
    return client_datasets, tokenized_datasets
