import csv
import os
import random
from protocolo.config import amostra, conjuntos, semente, tarefas

origens = {
    "carolina": "dados/carolina_soc.csv",
    "tweets": "dados/tweets_test.csv",
    "goemotions": "dados/goemotions_teste.csv",
}

os.makedirs("dados", exist_ok=True)

for conjunto in conjuntos:
    random.seed(semente)
    with open(origens[conjunto], encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))

    if conjunto == "goemotions":
        emocoes = tarefas["emocao"]["rotulos"]
        escolhidas = []
        for emocao in emocoes:
            da_emocao = [linha for linha in linhas if linha["rotulo"] == emocao]
            escolhidas += random.sample(da_emocao, min(amostra // len(emocoes), len(da_emocao)))
    else:
        escolhidas = random.sample(linhas, min(amostra, len(linhas)))

    saida = f"dados/amostra_{conjunto}.csv"
    with open(saida, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "texto", "gabarito"])
        for linha in escolhidas:
            writer.writerow([linha["id"], linha["texto"], linha.get("rotulo", "")])
    print(f"{len(escolhidas)} frases de {origens[conjunto]} em {saida}")
