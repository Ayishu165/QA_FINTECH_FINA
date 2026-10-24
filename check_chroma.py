import json

path=r"D:\inu\QA_FINTECH_FINAL\data\embeddings\chunk_embeddings.json"

with open(path,encoding="utf-8") as f:
    data=json.load(f)

print(type(data))

if isinstance(data,list):
    print("number of embeddings:",len(data))
    print(data[0].keys())

else:
    print(data.keys())