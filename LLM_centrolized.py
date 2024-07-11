import torch
import copy
import numpy as np
import random
import warnings
warnings.filterwarnings("ignore")

from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, GPT2ForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from peft import LoraConfig, get_peft_model

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 设置随机种子
set_seed(42)

accuracy_metric = load_metric("accuracy")

def predict_acc(trainer):
    
    accuracy_metric = load_metric("accuracy")
    predictions = trainer.predict(tokenized_datasets["validation"])
    pred = np.argmax(predictions.predictions, axis=-1)
    accuracy = accuracy_metric.compute(predictions=pred, references=predictions.label_ids)
    return accuracy


# 加載 IMDB 資料集
# dataset = load_dataset("imdb")
dataset = load_dataset("glue", "qnli")

# 初始化 GPT-2 的 tokenizer 並設置 padding token
tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 定義 tokenization 函數
def tokenize_function(examples):
    # return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
    return tokenizer(examples["question"], examples["sentence"], truncation=True, max_length=128)
    # return tokenizer(examples["sentence"], truncation=True, max_length=128)

# 對資料集進行 tokenization
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 移除無用的列，只保留 input_ids 和 labels
# tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.remove_columns(["question", "sentence", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

save_dir = "./llm_models/centrelized/qnli/"
client_num = 5
client_datasets = [{} for i in range(client_num)]
for data in tokenized_datasets:
    # 获取训练集
    data_dataset = tokenized_datasets[data]
    
    # 假设我们要将训练集拆分为两个子集
    split_ratio = 0.2
    split_index = int(len(data_dataset) * split_ratio)
    
    # 创建两个训练数据集
    for i in range(client_num):
        client_datasets[i][data] = data_dataset.select(range(i*split_index,(i+1)*split_index))


# 設置 LoRA 配置
lora_config = LoraConfig(
    r=8,  # Rank
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
)

# 訓練參數設置
training_args = TrainingArguments(
    output_dir=save_dir,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
)

# 加載 GPT-2 模型並設置 padding token id
model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=2)
model.config.pad_token_id = model.config.eos_token_id

# 使用 LoRA 配置模型
peft_model = get_peft_model(model, lora_config)
client_models = [copy.deepcopy(peft_model) for idx in range(client_num)]

acc_score = {}
# 預測未 fine-tune 的結果
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
)
acc = predict_acc(trainer)
acc_score["before"] = acc

# 對 LoRA 進行訓練
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
)
trainer.train()

save_dir = "./llm_models/centrelized/"
print(f"save pretrained model in {save_dir}")
peft_model.save_pretrained(save_dir)
    
# 預測未 fine-tune 的結果
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
)
acc = predict_acc(trainer)
acc_score["after"] = acc
before = acc_score["before"]
after = acc_score["after"]

print(f"accuracy before LoRA fine-tune : {before}\naccuracy after LoRA fine-tune : {after}")     