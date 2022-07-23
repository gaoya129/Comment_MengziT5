import json

def statistic(filepath):
    news_len = 0;min_news = 1000;max_news = 0
    review_len = 0;min_review = 1000;max_review = 0
    file = open(filepath, 'r', encoding='utf-8')
    data = []
    for line in file.readlines():
        dic = json.loads(line)
        nlen = len(dic['context'])
        rlen = len(dic['review'])
        news_len += nlen
        min_news = nlen if nlen < min_news else min_news
        max_news = nlen if nlen > max_news else max_news
        min_review = rlen if rlen < min_review else min_review
        max_review = rlen if rlen > max_review else max_review
        review_len += rlen
        data.append(dic)
    n = len(data)
    print(filepath,' news mean length:',news_len/n,' reviews mean length:',review_len/n)
    print('For NEWS: the max length is',max_news,'the min length is',min_news)
    print('For REVIEWS: the max length is',max_review,'the min length is',min_review)

if __name__ == "__main__":
    statistic('./data/etrain.json')
    statistic('./data/etest.json')
    statistic('./data/edev.json')
