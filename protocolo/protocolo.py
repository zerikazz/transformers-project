import csv
import os
import random
import torch
from protocolo.juizes import (
    carregar,
    carregar_lexico,
    emocoes,
    juiz_atencao,
    juiz_embeddings,
    juiz_nli,
    juiz_nli_llm,
    kappa_fleiss,
    voto,
)

familias = {
    "transformer": {
        "encoder": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "nli": "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
    },
    "deepseek": {
        "llm": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    },
}

familia = os.environ.get("FAMILIA", "transformer")
amostra = int(os.environ.get("AMOSTRA", 100))
caminho_lexico = os.environ.get("LEXICO", "dados/lexico.csv")
conjunto = os.environ.get("CONJUNTO", "carolina")
saida = f"saida/juizes_{familia}_{conjunto}.csv"
semente = 42

random.seed(semente)
torch.manual_seed(semente)

if conjunto == "goemotions":
    with open("dados/goemotions_teste.csv", encoding="utf-8") as f:
        todas = [linha for linha in csv.DictReader(f) if linha["rotulo"] in emocoes]
    linhas = []
    for emocao in emocoes:
        da_emocao = [linha for linha in todas if linha["rotulo"] == emocao]
        linhas += random.sample(da_emocao, min(amostra // len(emocoes), len(da_emocao)))
else:
    with open("saida/polaridade.csv", encoding="utf-8") as f:
        linhas = [
            linha for linha in csv.DictReader(f)
            if linha["bertweet_faixa"] != "neutro" and linha["roberta_faixa"] != "neutro"
        ]
    linhas = random.sample(linhas, min(amostra, len(linhas)))

tem_gabarito = conjunto == "goemotions"
textos = [linha["texto"] for linha in linhas]

lexico = carregar_lexico(caminho_lexico) if os.path.exists(caminho_lexico) else {}
if not lexico:
    print(f"Léxico não encontrado em {caminho_lexico}, o juiz de atenção vai se abster em todas as frases.\n")

modelos = familias[familia]

if familia == "deepseek":
    tokenizer, model = carregar(modelos["llm"], decoder=True)
    scores = {
        "embeddings": juiz_embeddings(tokenizer, model, textos),
        "nli": juiz_nli_llm(tokenizer, model, textos),
        "atencao": juiz_atencao(tokenizer, model, textos, lexico, decoder=True),
    }
else:
    tokenizer, model = carregar(modelos["encoder"])
    scores = {
        "embeddings": juiz_embeddings(tokenizer, model, textos),
        "nli": juiz_nli(modelos["nli"], textos),
        "atencao": juiz_atencao(tokenizer, model, textos, lexico),
    }

juizes = list(scores)
votos = [[voto(scores[juiz][i]) for juiz in juizes] for i in range(len(textos))]

os.makedirs("saida", exist_ok=True)

concordancias = []
rotulos_juizes = []
with open(saida, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    cabecalho = ["id", "texto"]
    for juiz in juizes:
        cabecalho += [f"{juiz}_{emocao}" for emocao in emocoes] + [f"{juiz}_voto"]
    cabecalho += ["rotulo_juizes", "concordancia", "divergente"]
    if tem_gabarito:
        cabecalho += ["rotulo_humano", "concordancia_humana"]
    writer.writerow(cabecalho)

    for i, linha in enumerate(linhas):
        validos = [v for v in votos[i] if v]
        rotulo = max(set(validos), key=validos.count) if validos else ""
        concordancia = validos.count(rotulo) if validos else 0
        concordancias.append(concordancia)

        registro = [linha["id"], linha["texto"]]
        for j, juiz in enumerate(juizes):
            registro += [f"{s:.4f}" for s in scores[juiz][i]] + [votos[i][j] or ""]
        registro += [rotulo if concordancia >= 2 else "", concordancia, int(concordancia < len(juizes))]
        if tem_gabarito:
            registro += [linha["rotulo"], linha["concordancia_humana"]]
        writer.writerow(registro)
        rotulos_juizes.append(rotulo)

print(f"--- TRÊS JUÍZES ({familia.upper()}) EM {len(textos)} FRASES DO {conjunto.upper()} ---\n")
for j, juiz in enumerate(juizes):
    contagem = {emocao: 0 for emocao in emocoes + [None]}
    for v in votos:
        contagem[v[j]] += 1
    print(f"{juiz}: " + " | ".join(f"{e or 'absteve'}: {c}" for e, c in contagem.items()))

print("-" * 60)
for n in range(len(juizes), -1, -1):
    print(f"{n} juízes concordam: {concordancias.count(n)} frases")
kappa, usadas = kappa_fleiss(votos)
print(f"Kappa de Fleiss: {kappa:.4f} (nas {usadas} frases em que nenhum juiz se absteve)")

if tem_gabarito:
    gabarito = [linha["rotulo"] for linha in linhas]
    humana = [float(linha["concordancia_humana"]) for linha in linhas]

    print("-" * 60)
    for j, juiz in enumerate(juizes):
        acertos = sum(v[j] == g for v, g in zip(votos, gabarito))
        print(f"Acurácia do juiz {juiz}: {acertos / len(gabarito) * 100:.2f}%")
    acertos = sum(r == g for r, g in zip(rotulos_juizes, gabarito))
    print(f"Acurácia da maioria: {acertos / len(gabarito) * 100:.2f}% (chance: {100 / len(emocoes):.2f}%)")

    print("-" * 60)
    print("Hipótese: quanto mais juízes concordam, mais o rótulo acerta")
    for n in range(len(juizes), 0, -1):
        indices = [i for i, c in enumerate(concordancias) if c == n]
        if not indices:
            continue
        acerto = sum(rotulos_juizes[i] == gabarito[i] for i in indices) / len(indices) * 100
        media_humana = sum(humana[i] for i in indices) / len(indices)
        print(f"  {n} juízes concordam: {len(indices)} frases, acerto {acerto:.2f}%, concordância humana média {media_humana:.3f}")

print(f"Resultado frase a frase em {saida}")
