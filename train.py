from transformers import T5Tokenizer, T5ForConditionalGeneration, DataCollatorWithPadding
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import torch
from datetime import datetime
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['PYTORCH_CUDA_ALLOC_CONF']=<max_split_size_mb>:<50>

def tokenize_func(data):
    Mengzi_tokenizer = T5Tokenizer.from_pretrained("Langboat/mengzi-t5-base")
    input_encodings = t5_tokenizer(data['context'],max_length=1024,truncation=True,padding='max_length')
    target_encodings = t5_tokenizer(data['review'],max_length=128,truncation=True,padding='max_length')
    return {"input_ids": input_encodings['input_ids'],
            "attention_mask": input_encodings["attention_mask"],
            "labels": target_encodings["input_ids"]}

if __name__ == "__main__":
    folder = "./data/"
    # 数据集
    t5_tokenizer = T5Tokenizer.from_pretrained("Langboat/mengzi-t5-base")
    all_data = load_dataset('json', data_files={'train':folder+'train2.json','test':folder+'test.json'})
    print(all_data)
    tokenized_datasets = all_data.map(tokenize_func,batched=True)
    # print(tokenized_datasets['train'][:3])
    # print(tokenized_datasets['test'][:3])
    print(type(tokenized_datasets))

    # 定义数据加载器
    data_collator = DataCollatorWithPadding(tokenizer=t5_tokenizer)
    tokenized_datasets = tokenized_datasets.remove_columns(["context", "review"])
    tokenized_datasets.set_format("torch")
    print(tokenized_datasets["train"].column_names)
    
    train_dataset = tokenized_datasets["train"].shuffle(seed=42)
    eval_dataset = tokenized_datasets["test"].shuffle(seed=42)
    
    

    # 模型
    model = T5ForConditionalGeneration.from_pretrained("Langboat/mengzi-t5-base", cache_dir='./t5')
    # 使用GPU
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    print(torch.cuda.memory_stats(device=device))
    
    save_dir_suffix = datetime.now().strftime("%Y%m%d%H%m%S")
    training_args = TrainingArguments(
        num_train_epochs=3,
        per_device_train_batch_size=2, # batch_size需要根据自己GPU的显存进行设置，2080,8G显存，batch_size设置为2可以跑起来。
        logging_steps=10,
        #fp16=True,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=50,
        load_best_model_at_end=True,
        learning_rate=1e-5,
        save_total_limit=3,
        #warmup_steps=100,
        output_dir="./ckpts/batch_size2_data2w",
        )
    
    trainer = Trainer(
        tokenizer=t5_tokenizer,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    torch.cuda.empty_cache()
    trainer.train()
    
    trainer.savemodel('model/'+str(save_dir_suffix))
    result = trainer.evaluate()
    print(result)