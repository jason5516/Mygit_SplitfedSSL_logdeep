import torch
import copy
import numpy as np
import random
import warnings
warnings.filterwarnings("ignore")

from utils.GLUE_partitaon_dataset import load_partition_glue_data
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
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

# data list = ["mrpc", "rte", "wnli", "qnli", "sst2"]
save_dir = "./llm_models/FL/"
client_num = 10
alpha = 1.0
data = "mrpc"
# global_epochs = 100
global_epochs = 10
accuracy_metric = load_metric("accuracy")

def predict_acc(trainer, test_data):
    
    accuracy_metric = load_metric("accuracy")
    predictions = trainer.predict(test_data)
    pred = np.argmax(predictions.predictions, axis=-1)
    accuracy = accuracy_metric.compute(predictions=pred, references=predictions.label_ids)
    return accuracy



client_datasets = load_partition_glue_data(data, client_num, alpha)


# 設置 LoRA 配置
lora_config = LoraConfig(
    r=16,  # Rank
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
)

# 訓練參數設置
training_args = TrainingArguments(
    output_dir=save_dir,
    evaluation_strategy="epoch",
    learning_rate=0.001,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=100,
    num_train_epochs=1,
    weight_decay=0.1,
    logging_dir="./logs",
)

# 加載 GPT-2 模型並設置 padding token id
# model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=2)
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
model.config.pad_token_id = model.config.eos_token_id

tokenizer = AutoTokenizer.from_pretrained("./llm_models/roberta_base")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 使用 LoRA 配置模型
peft_model = get_peft_model(model, lora_config)
client_models = [copy.deepcopy(peft_model) for idx in range(client_num)]

# acc_score_before = {}
# 預測未 fine-tune 的結果
# for idx, data in enumerate(data_list):
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=client_datasets[idx]["train"],
#         eval_dataset=client_datasets[idx]["valid"],
#         data_collator=data_collator,
#     )
#     acc = predict_acc(trainer, client_datasets[idx]["test"])
#     acc_score_before[data] = acc


client_weights = [1/client_num for i in range(client_num)]
client_acc = [{} for i in range(client_num)]
for epoch in range(global_epochs):
    print(f"------------epoch {epoch}------------")
    for c_idx, c_model in enumerate(client_models):
        print(f"-- client{c_idx} training--")
        
        trainer = Trainer(
            model=c_model,
            args=training_args,
            train_dataset=client_datasets[c_idx]["train"],
            eval_dataset=client_datasets[c_idx]["valid"],
            data_collator=data_collator,
        )
        
        trainer.train()

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
            eval_dataset=client_datasets[c_idx]["valid"],
            data_collator=data_collator,
        )


print(f"save pretrained model in {save_dir}")
peft_model.save_pretrained(save_dir)
    
acc_score_after = {}
# 預測未 fine-tune 的結果
for idx, data in enumerate(data_list):
# 預測未 fine-tune 的結果
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=client_datasets[idx]["train"],
        eval_dataset=client_datasets[idx]["valid"],
        data_collator=data_collator,
    )
    acc = predict_acc(trainer, client_datasets[idx]["test"])
    acc_score_after[data] = acc


for data in data_list:
    print(f"After LoRA fine-tune :\n")
    acc_after = acc_score_after[data]
    print(f"Data {data} : {acc_after}\n")