from datasets import load_dataset
from transformers import AutoTokenizer

# 初始化 GPT-2 的 tokenizer 並設置 padding token
tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

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

def load_glue_data(data = "mrpc"):
    # 加載 IMDB 資料集
    # "wnli" 有問題的資料集
    # glue dataset ["mrpc", "sst2", "rte", "mnli", "qnli", "qqp"]
    # dataset = load_dataset("imdb")
    dataset = load_dataset("glue", data)


    # 對資料集進行 tokenization
    if data == "mrpc":
        tokenized_datasets = dataset.map(mrpc_tokenize_function, batched=True)
        # 移除無用的列，只保留 input_ids 和 labels
        tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        tokenized_datasets.set_format("torch")
        return tokenized_datasets
    elif data == "sst2":
        tokenized_datasets = dataset.map(sst2_tokenize_function, batched=True)
        # 移除無用的列，只保留 input_ids 和 labels
        tokenized_datasets = tokenized_datasets.remove_columns(["sentence", "idx"])
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        tokenized_datasets.set_format("torch")
        return tokenized_datasets
    elif data == "rte":
        tokenized_datasets = dataset.map(rte_tokenize_function, batched=True)
        # 移除無用的列，只保留 input_ids 和 labels
        tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        tokenized_datasets.set_format("torch")
        return tokenized_datasets
    elif data == "mnli":
        tokenized_datasets = dataset.map(mnli_tokenize_function, batched=True)
        # 移除無用的列，只保留 input_ids 和 labels
        tokenized_datasets = tokenized_datasets.remove_columns(["premise", "hypothesis", "idx"])
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        tokenized_datasets.set_format("torch")
        return tokenized_datasets
    elif data == "qnli":
        tokenized_datasets = dataset.map(qnli_tokenize_function, batched=True)
        # 移除無用的列，只保留 input_ids 和 labels
        tokenized_datasets = tokenized_datasets.remove_columns(["question", "sentence", "idx"])
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        tokenized_datasets.set_format("torch")
        return tokenized_datasets
    elif data == "wnli":
        tokenized_datasets = dataset.map(wnli_tokenize_function, batched=True)
        # 移除無用的列，只保留 input_ids 和 labels
        tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        tokenized_datasets.set_format("torch")
        return tokenized_datasets
    elif data == "qqp":
        tokenized_datasets = dataset.map(qqp_tokenize_function, batched=True)
        # 移除無用的列，只保留 input_ids 和 labels
        tokenized_datasets = tokenized_datasets.remove_columns(["question1", "question2", "idx"])
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        tokenized_datasets.set_format("torch")
        return tokenized_datasets
    else:
        tokenized_datasets = dataset.map(mrpc_tokenize_function, batched=True)
        # 移除無用的列，只保留 input_ids 和 labels
        tokenized_datasets = tokenized_datasets.remove_columns(["text", "idx"])
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        tokenized_datasets.set_format("torch")
        return tokenized_datasets
