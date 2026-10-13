import csv
import os
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

classes = ["negativo", "neutro", "positivo"]
modelo_base = "neuralmind/bert-base-portuguese-cased"
saida = "saida/modelos/bertimbau-sentimento"
semente = 42
epocas = int(os.environ.get("EPOCAS", 4))
lote = 16
taxa = 2e-5

torch.manual_seed(semente)
np.random.seed(semente)
device = "cuda" if torch.cuda.is_available() else "cpu"


def ler(divisao):
    with open(f"dados/tweets_{divisao}.csv", encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))
    return [l["texto"] for l in linhas], torch.tensor([classes.index(l["rotulo"]) for l in linhas])


def lotes(textos, rotulos, embaralhar):
    ordem = torch.randperm(len(textos)) if embaralhar else torch.arange(len(textos))
    for i in range(0, len(textos), lote):
        indices = ordem[i:i + lote]
        inputs = tokenizer([textos[j] for j in indices], padding=True, truncation=True, max_length=128, return_tensors="pt")
        yield inputs.to(device), rotulos[indices].to(device)


def avaliar(textos, rotulos):
    model.eval()
    previstos = []
    with torch.no_grad():
        for inputs, _ in lotes(textos, rotulos, False):
            with torch.autocast(device, enabled=device == "cuda"):
                previstos.append(model(**inputs).logits.argmax(dim=1).cpu())
    model.train()
    return (torch.cat(previstos) == rotulos).float().mean().item() * 100


textos_treino, y_treino = ler("train")
textos_validacao, y_validacao = ler("validation")
textos_teste, y_teste = ler("test")

tokenizer = AutoTokenizer.from_pretrained(modelo_base)
model = AutoModelForSequenceClassification.from_pretrained(
    modelo_base,
    num_labels=len(classes),
    id2label=dict(enumerate(classes)),
    label2id={c: i for i, c in enumerate(classes)},
).to(device)

otimizador = torch.optim.AdamW(model.parameters(), lr=taxa, weight_decay=0.01)
passos = epocas * ((len(textos_treino) + lote - 1) // lote)
agenda = get_linear_schedule_with_warmup(otimizador, int(0.1 * passos), passos)
escala = torch.amp.GradScaler(enabled=device == "cuda")

print(f"--- FINE-TUNING DO BERTIMBAU EM {len(textos_treino)} TWEETS ---\n")
print(f"Antes do treino (cabeça aleatória), teste: {avaliar(textos_teste, y_teste):.2f}%")

melhor = -1.0
model.train()
for epoca in range(1, epocas + 1):
    for inputs, rotulos in lotes(textos_treino, y_treino, True):
        with torch.autocast(device, enabled=device == "cuda"):
            perda = model(**inputs, labels=rotulos).loss
        otimizador.zero_grad()
        escala.scale(perda).backward()
        escala.step(otimizador)
        escala.update()
        agenda.step()

    acuracia = avaliar(textos_validacao, y_validacao)
    print(f"Época {epoca}: validação {acuracia:.2f}%")
    if acuracia > melhor:
        melhor = acuracia
        model.save_pretrained(saida)
        tokenizer.save_pretrained(saida)

model = AutoModelForSequenceClassification.from_pretrained(saida).to(device)
print(f"Melhor época na validação: {melhor:.2f}%")
print(f"Depois do fine-tuning, teste: {avaliar(textos_teste, y_teste):.2f}%")
print(f"Modelo salvo em {saida}")
