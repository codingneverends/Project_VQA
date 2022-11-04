import json

train_qns = open('../Dataset/Slake/train.json',encoding="utf8")
train_qns = json.load(train_qns)

train_qns_en = [x['answer'] for x in train_qns if x['q_lang'] == 'en' and x['content_type']=='Modality']

print(len(train_qns_en))

print(set(train_qns_en))