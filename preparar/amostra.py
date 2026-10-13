import csv
import os
import random
from protocolo.config import amostra, calibracao, conjuntos, semente, tarefas

origens = {
    "carolina": "dados/carolina_soc.csv",
    "tweets": "dados/tweets_test.csv",
    "goemotions": "dados/goemotions_teste.csv",
}


def sortear(caminho, quantidade, por_rotulo=None):
    random.seed(semente)
    with open(caminho, encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))
    if not por_rotulo:
        return random.sample(linhas, min(quantidade, len(linhas)))
    escolhidas = []
    for rotulo in por_rotulo:
        do_rotulo = [linha for linha in linhas if linha["rotulo"] == rotulo]
        escolhidas += random.sample(do_rotulo, min(quantidade // len(por_rotulo), len(do_rotulo)))
    return escolhidas


def salvar(escolhidas, saida, origem):
    with open(saida, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "texto", "gabarito"])
        for linha in escolhidas:
            writer.writerow([linha["id"], linha["texto"], linha.get("rotulo", "")])
    print(f"{len(escolhidas)} frases de {origem} em {saida}")


os.makedirs("dados", exist_ok=True)
emocoes = tarefas["emocao"]["rotulos"]

for conjunto in conjuntos:
    escolhidas = sortear(origens[conjunto], amostra, emocoes if conjunto == "goemotions" else None)
    salvar(escolhidas, f"dados/amostra_{conjunto}.csv", origens[conjunto])

for nome, tarefa in tarefas.items():
    escolhidas = sortear(tarefa["treino"], calibracao, emocoes if nome == "emocao" else None)
    salvar(escolhidas, f"dados/calibracao_{nome}.csv", tarefa["treino"])
