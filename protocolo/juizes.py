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

emocoes = ["alegria", "tristeza", "raiva", "medo", "surpresa", "nojo"]

prototipos = {
    "alegria": ["Estou muito feliz hoje!", "Que notícia maravilhosa, adorei!", "Isso me deixou radiante."],
    "tristeza": ["Estou muito triste.", "Que saudade, meu coração está partido.", "Isso me deixou para baixo."],
    "raiva": ["Estou com muita raiva!", "Que ódio, isso é um absurdo!", "Fiquei furioso com isso."],
    "medo": ["Estou com muito medo.", "Isso me deixa apavorado.", "Tenho pavor do que pode acontecer."],
    "surpresa": ["Nossa, não acredito!", "Que surpresa, eu não esperava por isso!", "Fiquei chocado com a novidade."],
    "nojo": ["Que nojo!", "Isso é repugnante.", "Senti nojo daquilo."],
}

hipotese = "Este texto expressa {}."

resumo_llm = "Texto: \"{texto}\"\nEm uma palavra, a emoção deste texto é:"

neutros_llm = ["N/A", "", "[sem texto]"]

pergunta_llm = "Texto: \"{texto}\"\nPergunta: Este texto expressa {emocao}? Responda apenas sim ou não.\nResposta:"

device = "cuda" if torch.cuda.is_available() else "cpu"


def normalizar(palavra):
    sem_acento = unicodedata.normalize("NFD", palavra.lower())
    return "".join(c for c in sem_acento if unicodedata.category(c) != "Mn")


def carregar_lexico(caminho):
    lexico = {}
    with open(caminho, encoding="utf-8") as f:
        for linha in csv.DictReader(f):
            if linha["emocao"] in emocoes:
                lexico.setdefault(normalizar(linha["palavra"]), set()).add(linha["emocao"])
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


def embeddings(tokenizer, model, textos, decoder=False):
    vetores = []
    for texto in textos:
        entrada = resumo_llm.format(texto=texto) if decoder else texto
        inputs = tokenizer(entrada, truncation=True, max_length=160, return_tensors="pt").to(model.device)
        with torch.no_grad():
            estados = model(**inputs, output_hidden_states=True).hidden_states[-1][0].float()
        vetores.append(F.normalize(estados[-1] if decoder else estados.mean(dim=0), dim=0))
    return torch.stack(vetores)


def juiz_embeddings(tokenizer, model, textos, decoder=False):
    centros = torch.stack([
        F.normalize(embeddings(tokenizer, model, prototipos[emocao], decoder).mean(dim=0), dim=0)
        for emocao in emocoes
    ])
    return (embeddings(tokenizer, model, textos, decoder) @ centros.T).cpu().tolist()


def juiz_nli(nome_modelo, textos):
    tokenizer = AutoTokenizer.from_pretrained(nome_modelo)
    model = AutoModelForSequenceClassification.from_pretrained(nome_modelo).to(device)
    model.eval()
    entailment = {r.lower(): i for i, r in model.config.id2label.items()}["entailment"]

    scores = []
    for texto in textos:
        inputs = tokenizer(
            [texto] * len(emocoes),
            [hipotese.format(emocao) for emocao in emocoes],
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        scores.append(F.softmax(logits, dim=-1)[:, entailment].cpu().tolist())
    return scores


def juiz_nli_llm(tokenizer, model, textos):
    sim = tokenizer.encode(" sim", add_special_tokens=False)[0]
    nao = tokenizer.encode(" não", add_special_tokens=False)[0]

    def chance_sim(texto, emocao):
        inputs = tokenizer(pergunta_llm.format(texto=texto, emocao=emocao), return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1].float()
        return (logits[sim] - logits[nao]).item()

    base = [sum(chance_sim(neutro, emocao) for neutro in neutros_llm) / len(neutros_llm) for emocao in emocoes]
    return [[chance_sim(texto, emocao) - base[i] for i, emocao in enumerate(emocoes)] for texto in textos]


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


def juiz_atencao(tokenizer, model, textos, lexico, decoder=False):
    scores = []
    for texto in textos:
        pesos = atencao_por_palavra(tokenizer, model, texto, decoder)
        linha = [0.0] * len(emocoes)
        for palavra, peso in pesos.items():
            for emocao in lexico.get(palavra, ()):
                linha[emocoes.index(emocao)] += peso
        scores.append(linha)
    return scores


def voto(scores):
    if max(scores) <= 0:
        return None
    return emocoes[max(range(len(scores)), key=lambda i: scores[i])]


def kappa_fleiss(votos):
    validos = [v for v in votos if None not in v]
    if len(validos) < 2:
        return float("nan"), len(validos)

    n = len(validos[0])
    contagens = [[v.count(emocao) for emocao in emocoes] for v in validos]
    p_i = [(sum(c * c for c in linha) - n) / (n * (n - 1)) for linha in contagens]
    p_j = [sum(linha[j] for linha in contagens) / (len(validos) * n) for j in range(len(emocoes))]

    p_barra = sum(p_i) / len(p_i)
    p_e = sum(p * p for p in p_j)
    if p_e == 1:
        return 1.0, len(validos)
    return (p_barra - p_e) / (1 - p_e), len(validos)
