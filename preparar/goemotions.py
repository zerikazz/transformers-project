import csv
import hashlib
import os
from collections import Counter, defaultdict
from huggingface_hub import hf_hub_download

repo_goemotions = "antoniomenezes/go_emotions_ptbr"
arquivos = ["goemotions_1_pt.csv", "goemotions_2_pt.csv", "goemotions_3_pt.csv"]

ekman = {
    "raiva": ["anger", "annoyance", "disapproval"],
    "nojo": ["disgust"],
    "medo": ["fear", "nervousness"],
    "alegria": [
        "joy", "amusement", "approval", "excitement", "gratitude", "love",
        "optimism", "relief", "pride", "admiration", "desire", "caring",
    ],
    "tristeza": ["sadness", "disappointment", "embarrassment", "grief", "remorse"],
    "surpresa": ["surprise", "realization", "confusion", "curiosity"],
    "neutro": ["neutral"],
}

min_avaliadores = 3
fracao_teste = 0.2

os.makedirs("dados", exist_ok=True)

textos = {}
votos = defaultdict(list)

for arquivo in arquivos:
    caminho = hf_hub_download(repo_goemotions, arquivo, repo_type="dataset")
    with open(caminho, encoding="utf-8") as f:
        for linha in csv.DictReader(f):
            if linha["example_very_unclear"] == "True":
                continue
            marcadas = {
                emocao for emocao, originais in ekman.items()
                if any(linha[original] == "1" for original in originais)
            }
            if len(marcadas) != 1:
                continue
            textos[linha["id"]] = " ".join(linha["texto"].split())
            votos[linha["id"]].append(marcadas.pop())

totais = Counter()
for divisao in ["treino", "teste"]:
    saida = f"dados/goemotions_{divisao}.csv"
    with open(saida, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "texto", "rotulo", "avaliadores", "concordancia_humana"])

        for id_texto, lista in votos.items():
            if len(lista) < min_avaliadores:
                continue
            sorteio = int(hashlib.md5(id_texto.encode()).hexdigest(), 16) % 100 / 100
            if (sorteio < fracao_teste) != (divisao == "teste"):
                continue

            rotulo, vezes = Counter(lista).most_common(1)[0]
            writer.writerow([id_texto, textos[id_texto], rotulo, len(lista), f"{vezes / len(lista):.4f}"])
            totais[divisao] += 1

    print(f"{totais[divisao]} frases do GoEmotions ({divisao}) salvas em {saida}")
