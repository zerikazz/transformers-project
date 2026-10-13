import csv
import re
import unicodedata
import torch
import torch.nn.functional as F
import os
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from protocolo.config import neutros_llm

device = "cuda" if torch.cuda.is_available() else "cpu"


def normalizar(palavra):
    sem_acento = unicodedata.normalize("NFD", palavra.lower())
    return "".join(c for c in sem_acento if unicodedata.category(c) != "Mn")


def carregar_lexico(caminho, rotulos):
    lexico = {}
    with open(caminho, encoding="utf-8") as f:
        for linha in csv.DictReader(f):
            if linha["rotulo"] in rotulos:
                lexico.setdefault(normalizar(linha["palavra"]), set()).add(linha["rotulo"])
    return lexico


def carregar(nome_modelo, decoder=False):
    tokenizer = AutoTokenizer.from_pretrained(nome_modelo)
    classe = AutoModelForCausalLM if decoder else AutoModel
    if os.environ.get("QUANTIZAR") == "4bit":
        quantizacao = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        model = classe.from_pretrained(
            nome_modelo, quantization_config=quantizacao, device_map="auto", attn_implementation="eager"
        )
    else:
        model = classe.from_pretrained(nome_modelo, dtype="auto", attn_implementation="eager").to(device)
    model.eval()
    return tokenizer, model


def embeddings(tokenizer, model, textos, resumo=None):
    vetores = []
    for texto in textos:
        entrada = resumo.format(texto=texto) if resumo else texto
        inputs = tokenizer(entrada, truncation=True, max_length=160, return_tensors="pt").to(model.device)
        with torch.no_grad():
            estados = model(**inputs, output_hidden_states=True).hidden_states[-1][0].float()
        vetores.append(F.normalize(estados[-1] if resumo else estados.mean(dim=0), dim=0))
    return torch.stack(vetores)


def juiz_embeddings(tokenizer, model, textos, prototipos, resumo=None):
    centros = torch.stack([
        F.normalize(embeddings(tokenizer, model, frases, resumo).mean(dim=0), dim=0)
        for frases in prototipos.values()
    ])
    return (embeddings(tokenizer, model, textos, resumo) @ centros.T).cpu().tolist()


def juiz_nli(nome_modelo, textos, rotulos, hipotese):
    tokenizer = AutoTokenizer.from_pretrained(nome_modelo)
    model = AutoModelForSequenceClassification.from_pretrained(nome_modelo).to(device)
    model.eval()
    entailment = {r.lower(): i for i, r in model.config.id2label.items()}["entailment"]

    scores = []
    for texto in textos:
        inputs = tokenizer(
            [texto] * len(rotulos),
            [hipotese.format(rotulo) for rotulo in rotulos],
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        scores.append(F.softmax(logits, dim=-1)[:, entailment].cpu().tolist())
    return scores


def juiz_nli_llm(tokenizer, model, textos, rotulos, pergunta):
    sim = tokenizer.encode(" sim", add_special_tokens=False)[0]
    nao = tokenizer.encode(" não", add_special_tokens=False)[0]

    def chance_sim(texto, rotulo):
        inputs = tokenizer(pergunta.format(texto=texto, rotulo=rotulo), return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1].float()
        return (logits[sim] - logits[nao]).item()

    base = [sum(chance_sim(neutro, rotulo) for neutro in neutros_llm) / len(neutros_llm) for rotulo in rotulos]
    return [[chance_sim(texto, rotulo) - base[i] for i, rotulo in enumerate(rotulos)] for texto in textos]


def atencao_por_palavra(tokenizer, model, texto, decoder):
    inputs = tokenizer(texto, truncation=True, max_length=128, return_offsets_mapping=True, return_tensors="pt")
    offsets = inputs.pop("offset_mapping")[0].tolist()
    inputs = inputs.to(model.device)

    with torch.no_grad():
        atencao = model(**inputs, output_attentions=True).attentions[-1][0].float().mean(dim=0)

    tamanho = atencao.shape[0]
    if decoder:
        recebida = atencao.sum(dim=0) / torch.arange(tamanho, 0, -1, device=atencao.device)
    else:
        recebida = atencao.mean(dim=0)

    palavras = [(m.start(), m.end(), normalizar(m.group())) for m in re.finditer(r"\w+", texto)]
    pesos = {}
    for (inicio, fim), peso in zip(offsets, recebida.cpu().tolist()):
        if inicio == fim:
            continue
        for p_inicio, p_fim, palavra in palavras:
            if inicio < p_fim and fim > p_inicio:
                pesos[palavra] = pesos.get(palavra, 0.0) + peso
                break

    total = sum(pesos.values()) or 1.0
    return {palavra: peso / total for palavra, peso in pesos.items()}


def juiz_atencao(tokenizer, model, textos, lexico, rotulos, decoder=False):
    scores = []
    for texto in textos:
        pesos = atencao_por_palavra(tokenizer, model, texto, decoder)
        linha = [0.0] * len(rotulos)
        for palavra, peso in pesos.items():
            for rotulo in lexico.get(palavra, ()):
                linha[rotulos.index(rotulo)] += peso
        scores.append(linha)
    return scores


def voto(scores, rotulos):
    if max(scores) <= 0:
        return None
    return rotulos[max(range(len(scores)), key=lambda i: scores[i])]


def kappa_fleiss(votos, rotulos):
    validos = [v for v in votos if None not in v]
    if len(validos) < 2:
        return float("nan"), len(validos)

    n = len(validos[0])
    contagens = [[v.count(rotulo) for rotulo in rotulos] for v in validos]
    p_i = [(sum(c * c for c in linha) - n) / (n * (n - 1)) for linha in contagens]
    p_j = [sum(linha[j] for linha in contagens) / (len(validos) * n) for j in range(len(rotulos))]

    p_barra = sum(p_i) / len(p_i)
    p_e = sum(p * p for p in p_j)
    if p_e == 1:
        return 1.0, len(validos)
    return (p_barra - p_e) / (1 - p_e), len(validos)
