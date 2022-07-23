---
language: 

  - zh

license: apache-2.0

datasets:

- TencentKuaibao

metrics:

- bleu

- rouge
---
## 模型

- 基于中文[MengziT5](https://huggingface.co/Langboat/mengzi-t5-base)的新闻评论生成模型
- 数据集来源于论文[《Coherent Comment Generation for Chinese Articles with a Graph-to-Sequence Model》](https://github.com/lancopku/Graph-to-seq-comment-generation)

## 生成评论

- 在线API只能生成一种评论，模型通过设置model.generate()参数是可以生成多种评论的

```Python


t5_tokenizer = T5Tokenizer.from_pretrained("Langboat/mengzi-t5-base")


model = T5ForConditionalGeneration.from_pretrained("wawaup/MengziT5-Comment")


def generate_comment(input_ids):

    outputs = model.generate(input_ids,

                            max_length=128,

                            do_sample=True,

                            temperature=0.9,

                            early_stopping=True,

                            repetition_penalty=10.0,

                            top_p=0.5)

    print(outputs) 

    preds_cleaned = [t5_tokenizer.decode(ids, skip_special_tokens=True, 

                            clean_up_tokenization_spaces=True) for ids in outputs]

    print(preds_cleaned)

    return preds_cleaned

```
