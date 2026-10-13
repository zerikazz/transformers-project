import csv
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

modelos = {
    "bertweet": "pysentimiento/bertweet-pt-sentiment",
    "roberta": "cardiffnlp/xlm-roberta-base-tweet-sentiment-pt",
}

faixas = [
    (-1.0, "muito negativo"),
    (-0.6, "negativo"),
    (-0.2, "neutro"),
    (0.2, "positivo"),
    (0.6, "muito positivo"),
]

entrada = os.environ.get("ENTRADA", "dados/carolina_soc.csv")
saida = os.environ.get("SAIDA", "saida/polaridade.csv")
bertimbau = "saida/modelos/bertimbau-sentimento"

if os.path.isdir(bertimbau):
    modelos["bertimbau"] = bertimbau
amostra = int(os.environ.get("AMOSTRA", 500))
semente = 42
lote = 32

random.seed(semente)
torch.manual_seed(semente)
device = "cuda" if torch.cuda.is_available() else "cpu"


def faixa_do_score(score):
    nome = faixas[0][1]
    for limite, rotulo in faixas:
        if score >= limite:
            nome = rotulo
    return nome


def classificar(nome_modelo, textos):
    tokenizer = AutoTokenizer.from_pretrained(nome_modelo)
    model = AutoModelForSequenceClassification.from_pretrained(nome_modelo).to(device)
    model.eval()

    probabilidades = []
    for i in range(0, len(textos), lote):
        inputs = tokenizer(
            textos[i:i + lote],
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        probabilidades.append(F.softmax(logits, dim=-1).cpu())

    return torch.cat(probabilidades).numpy()


with open(entrada, encoding="utf-8") as f:
    linhas = list(csv.DictReader(f))

linhas = random.sample(linhas, min(amostra, len(linhas)))
textos = [linha["texto"] for linha in linhas]

resultados = {nome: classificar(modelo, textos) for nome, modelo in modelos.items()}

os.makedirs("saida", exist_ok=True)

with open(saida, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    cabecalho = ["id", "texto"]
    for nome in modelos:
        cabecalho += [f"{nome}_neg", f"{nome}_neu", f"{nome}_pos", f"{nome}_score", f"{nome}_faixa"]
    writer.writerow(cabecalho)

    for i, linha in enumerate(linhas):
        registro = [linha["id"], linha["texto"]]
        for nome in modelos:
            neg, neu, pos = resultados[nome][i]
            score = pos - neg
            registro += [f"{neg:.4f}", f"{neu:.4f}", f"{pos:.4f}", f"{score:.4f}", faixa_do_score(score)]
        writer.writerow(registro)

scores = {nome: resultados[nome][:, 2] - resultados[nome][:, 0] for nome in modelos}
rotulos = {nome: resultados[nome].argmax(axis=1) for nome in modelos}
nomes = list(modelos)

print(f"--- POLARIDADE EM {len(textos)} FRASES DE {entrada} ---\n")
for nome in nomes:
    contagem = {rotulo: 0 for _, rotulo in faixas}
    for score in scores[nome]:
        contagem[faixa_do_score(score)] += 1
    quantis = np.quantile(scores[nome], [0.2, 0.4, 0.6, 0.8])

    print(nome)
    print("  frases por faixa linear: " + " | ".join(f"{r}: {c}" for r, c in contagem.items()))
    print("  cortes se as faixas tivessem o mesmo número de frases: " + " | ".join(f"{q:.2f}" for q in quantis))

    if "rotulo" in linhas[0]:
        classes = ["negativo", "neutro", "positivo"]
        gabarito = np.array([classes.index(linha["rotulo"]) for linha in linhas])
        acuracia = (rotulos[nome] == gabarito).mean() * 100
        print(f"  acurácia contra o rótulo humano: {acuracia:.2f}%")
        for i, classe in enumerate(classes):
            media = scores[nome][gabarito == i].mean()
            print(f"  score médio das frases {classe}s: {media:+.3f}")
    print("-" * 60)

for a in range(len(nomes)):
    for b in range(a + 1, len(nomes)):
        concordancia = (rotulos[nomes[a]] == rotulos[nomes[b]]).mean() * 100
        correlacao = np.corrcoef(scores[nomes[a]], scores[nomes[b]])[0, 1]
        print(f"{nomes[a]} x {nomes[b]}: mesmo rótulo em {concordancia:.2f}%, correlação dos scores {correlacao:.4f}")
print(f"Resultado frase a frase em {saida}")
