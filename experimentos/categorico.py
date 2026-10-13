import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from transformers import AutoModel, AutoTokenizer

classes = ["negativo", "neutro", "positivo"]
modelo_base = "neuralmind/bert-base-portuguese-cased"
semente = 42
epocas = 300
corte = 1 / 3

torch.manual_seed(semente)
np.random.seed(semente)
device = "cuda" if torch.cuda.is_available() else "cpu"


def ler(divisao):
    with open(f"dados/tweets_{divisao}.csv", encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))
    return [l["texto"] for l in linhas], np.array([classes.index(l["rotulo"]) for l in linhas])


def embeddings(tokenizer, model, textos):
    vetores = []
    for i in range(0, len(textos), 32):
        inputs = tokenizer(textos[i:i + 32], padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            estados = model(**inputs).last_hidden_state
        mascara = inputs["attention_mask"].unsqueeze(-1)
        vetores.append(((estados * mascara).sum(dim=1) / mascara.sum(dim=1)).cpu())
    return torch.cat(vetores)


def treinar(rede, entrada, alvo, perda):
    otimizador = torch.optim.Adam(rede.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epocas):
        otimizador.zero_grad()
        perda(rede(entrada), alvo).backward()
        otimizador.step()
    rede.eval()
    return rede


textos_treino, y_treino = ler("train")
textos_teste, y_teste = ler("test")

print(f"--- CATEGÓRICO X CONTÍNUO EM {len(textos_teste)} TWEETS DE TESTE ---\n")

vetorizador = CountVectorizer(binary=True, strip_accents="unicode", token_pattern=r"(?u)\b[^\W\d_]{2,}\b", min_df=2)
x_treino = vetorizador.fit_transform(textos_treino)
x_teste = vetorizador.transform(textos_teste)
arvore = DecisionTreeClassifier(criterion="entropy", min_samples_leaf=3, random_state=semente).fit(x_treino, y_treino)
acuracia_arvore = (arvore.predict(x_teste) == y_teste).mean() * 100
palavras = vetorizador.get_feature_names_out()
mais_usadas = np.argsort(-arvore.feature_importances_)[:15]
print(f"Árvore de decisão por entropia nas palavras: {acuracia_arvore:.2f}%")
print("  palavras que a árvore mais usa para cortar: " + ", ".join(palavras[i] for i in mais_usadas))

tokenizer = AutoTokenizer.from_pretrained(modelo_base)
bertimbau = AutoModel.from_pretrained(modelo_base).to(device)
bertimbau.eval()
e_treino = embeddings(tokenizer, bertimbau, textos_treino).to(device)
e_teste = embeddings(tokenizer, bertimbau, textos_teste).to(device)
media, desvio = e_treino.mean(dim=0), e_treino.std(dim=0) + 1e-6
e_treino, e_teste = (e_treino - media) / desvio, (e_teste - media) / desvio

categorica = treinar(
    nn.Sequential(nn.Linear(768, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 3)).to(device),
    e_treino, torch.tensor(y_treino, device=device), F.cross_entropy,
)
with torch.no_grad():
    previsto = categorica(e_teste).argmax(dim=1).cpu().numpy()
acuracia_categorica = (previsto == y_teste).mean() * 100
print(f"Rede categórica (softmax em 3 classes) sobre o BERTimbau congelado: {acuracia_categorica:.2f}%")

continua = treinar(
    nn.Sequential(nn.Linear(768, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1), nn.Tanh()).to(device),
    e_treino, torch.tensor(y_treino - 1, dtype=torch.float32, device=device).unsqueeze(1), F.mse_loss,
)
with torch.no_grad():
    score = continua(e_teste).squeeze(1).cpu().numpy()
previsto = np.where(score <= -corte, 0, np.where(score >= corte, 2, 1))
acuracia_continua = (previsto == y_teste).mean() * 100
print(f"Rede contínua (tanh de -1 a 1, cortada em ±1/3) sobre o BERTimbau congelado: {acuracia_continua:.2f}%")

fatias = np.histogram(score, bins=[-1.0001, -0.6, -0.2, 0.2, 0.6, 1.0001])[0] / len(score) * 100
print("  % do score da rede contínua por faixa de 0,4: " + " | ".join(f"{f:.1f}" for f in fatias))
for i, classe in enumerate(classes):
    print(f"  score médio das frases {classe}s: {score[y_teste == i].mean():+.3f}")
