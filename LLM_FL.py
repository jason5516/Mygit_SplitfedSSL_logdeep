import torch
import copy
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, GPT2ForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from peft import LoraConfig, get_peft_model

accuracy_metric = load_metric("accuracy")

def predict_acc(trainer):
    
    accuracy_metric = load_metric("accuracy")
    predictions = trainer.predict(tokenized_datasets["test"])
    pred = np.argmax(predictions.predictions, axis=-1)
    accuracy = accuracy_metric.compute(predictions=pred, references=predictions.label_ids)
    return accuracy


# 加載 IMDB 資料集
# dataset = load_dataset("imdb")
dataset = load_dataset("glue", "mrpc")

# 初始化 GPT-2 的 tokenizer 並設置 padding token
tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 定義 tokenization 函數
def tokenize_function(examples):
    # return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=128)

# 對資料集進行 tokenization
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 移除無用的列，只保留 input_ids 和 labels
# tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

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
    output_dir="./llm_models/gpt2_LoRA/server",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
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
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
)
acc = predict_acc(trainer)
acc_score["before"] = acc


client_weights = [1/client_num for i in range(client_num)]
global_epochs = 2

for epoch in range(global_epochs):
    print(f"------------epoch {epoch}------------")
    for c_idx, c_model in enumerate(client_models):
        print(f"-- client{c_idx} training--")
        
        trainer = Trainer(
            model=c_model,
            args=training_args,
            train_dataset=client_datasets[c_idx]["train"],
            eval_dataset=client_datasets[c_idx]["test"],
            data_collator=data_collator,
        )
        
        trainer.train()
        # trainer.evaluate()
        # evaluation_results = trainer.evaluate()
        # print(evaluation_results)

    # aggregate model trainable parameters(adapter)
    with torch.no_grad():
        for key, param in peft_model.named_parameters():
            if param.requires_grad:
                temp = torch.zeros_like(param).cuda()
                for client_idx in range(client_num):
                    temp += client_weights[client_idx] * client_models[client_idx].state_dict()[key]                 
                peft_model.state_dict()[key].data.copy_(temp)
                for client_idx in range(client_num):
                    client_models[client_idx].state_dict()[key].data.copy_(peft_model.state_dict()[key])
    
    for c_idx, c_model in enumerate(client_models):
        trainer = Trainer(
            model=c_model,
            args=training_args,
            train_dataset=client_datasets[c_idx]["train"],
            eval_dataset=client_datasets[c_idx]["test"],
            data_collator=data_collator,
        )
    
# 預測未 fine-tune 的結果
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
)
acc = predict_acc(trainer)
acc_score["after"] = acc
before = acc_score["before"]
after = acc_score["after"]

print(f"accuracy before LoRA fine-tune : {before}\naccuracy after LoRA fine-tune : {after}")     