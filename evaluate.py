from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import json
import jieba
import numpy as np

def readjson(path):
    file = open(path, 'r', encoding='utf-8')
    data = []
    for line in file.readlines():
        dic = json.loads(line)
        data.append(dic)
    return data

def writejson(filepath,file):
    with open(filepath,"w",encoding="UTF-8") as f:
        for item in file:
            data = json.dump(item,fp=f,ensure_ascii=False)
            f.write('\n')
    print(filepath," wrtie finished!")



if __name__ == "__main__":
    RESULT_PATH = 'result/result.txt'
    TEST_PATH = 'data/etest.json'

    with open(RESULT_PATH, 'r') as txt:
        result_all = txt.readlines()

    test_all = readjson(TEST_PATH)
    test_rcut = []; test_review = []
    result_rcut = []; result_review = []
    for test_piece, result_piece in zip(test_all, result_all):
        tcut = ' '.join(jieba.lcut(test_piece['review']))
        test_rcut.append(tcut)
        test_review.append(test_piece['review'])

        rcut = ' '.join(jieba.lcut())
        result_rcut.append(rcut)
        result_review.append(result_piece)

    rouge = Rouge()
    report = rouge.get_scores(
        result_rcut, 
        test_rcut, 
        avg=True
    )
    print(report)

    smooth = SmoothingFunction()
    bleus = []
    for test_piece, result_piece in zip(test_review, result_review):
        score = sentence_bleu([test_piece], result_piece, smoothing_function=smooth.method1)
        bleus.append(score)
        # print(score)
        # break
    print(np.array(bleus).mean())

    