from transformers import T5Tokenizer, T5ForConditionalGeneration, DataCollatorWithPadding
import torch
import numpy as np
import json
from datasets import load_dataset
from datetime import datetime
from tqdm import tqdm

def tokenize_func(data):
    t5_tokenizer = T5Tokenizer.from_pretrained("Langboat/mengzi-t5-base")
    input_encodings = t5_tokenizer(data,max_length=1024,truncation=True,padding='max_length',return_tensors='pt')
    # target_encodings = t5_tokenizer(data['review'],max_length=256,truncation=True,padding='max_length',return_tensors='pt')
    return input_encodings['input_ids']

def readjson(path):
    file = open(path, 'r', encoding='utf-8')
    data = []
    for line in file.readlines():
        dic = json.loads(line)
        data.append(dic['context'])
    return data

def writejson(filepath,file):
    with open(filepath,"w",encoding="UTF-8") as f:
        for item in file:
            data = json.dump(item,fp=f,ensure_ascii=False,indent=2)
            f.write('\n')
    print(filepath," wrtie finished!")

def writetxt(filepath,file):
    with open(filepath,"w",encoding="UTF-8") as f:
        for item in file:
            f.write(item)
            f.write('\n')
    print(filepath," wrtie finished!")

def generate_comment(input_ids,cnt_num):
    outputs = model.generate(input_ids,
                            max_length=128,
                            do_sample=True,
                            temperature=0.9,
                            early_stopping=True,
                            repetition_penalty=10.0,
                            top_p=0.5,
                            num_return_sequences=cnt_num)
    print(outputs) 
    preds_cleaned = [t5_tokenizer.decode(ids, skip_special_tokens=True, 
                            clean_up_tokenization_spaces=True) for ids in outputs]
    print(preds_cleaned)
    return preds_cleaned

if __name__ == "__main__":
    start_time = datetime.now()
    t5_tokenizer = T5Tokenizer.from_pretrained("Langboat/mengzi-t5-base")
    
    file = './test/sample1.json'
    raw_data = readjson(file)
    tokenized = torch.stack([tokenize_func(item) for item in raw_data])
    print(tokenized)
    # tokenized.to(device)
    
    model = T5ForConditionalGeneration.from_pretrained("./ckpts/data16w/checkpoint-50000")
    
    cnt_num = 3
    # model.to(device)
    model.eval()
    print("model and tokenizer loaded.")
    news_comment = []
    reviews = []
    progress_bar = tqdm(range(len(raw_data)))
    for i,input_ids in enumerate(tokenized):
        print("start inference sentence ",i)
        # input_ids.to(device)
        print(input_ids)
        comment = generate_comment(input_ids,cnt_num)
        progress_bar.update(1)
        reviews.append(comment)
        # news_comment.append({'context':raw_data[i],"comment":comment})
    # writejson("./test/4wcomment13000-1.json",news_comment)
    writetxt('./test.txt',reviews)
    end_time = datetime.now()
    print("Inference cost time:",(end_time-start_time).seconds)

