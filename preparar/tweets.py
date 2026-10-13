import csv
import gzip
import json
import os
from huggingface_hub import hf_hub_download

repo_tweets = "mteb/tweet_sentiment_multilingual"
rotulos = {"0": "negativo", "1": "neutro", "2": "positivo"}

os.makedirs("dados", exist_ok=True)

for divisao in ["train", "validation", "test"]:
    caminho = hf_hub_download(repo_tweets, f"{divisao}/portuguese.jsonl.gz", repo_type="dataset")
    saida = f"dados/tweets_{divisao}.csv"

    with gzip.open(caminho, "rt", encoding="utf-8") as entrada, open(saida, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "texto", "rotulo"])
        total = 0
        for i, linha in enumerate(entrada):
            tweet = json.loads(linha)
            writer.writerow([i, " ".join(tweet["text"].split()), rotulos[str(tweet["label"])]])
            total += 1

    print(f"{total} tweets rotulados ({divisao}) salvos em {saida}")
