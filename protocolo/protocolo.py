import csv
import os
import random
import torch
from protocolo.config import conjuntos, familias, semente, tarefas, versao
from protocolo.juizes import (
    carregar,
    carregar_lexico,
    juiz_atencao,
    juiz_embeddings,
    juiz_lexico,
    juiz_nli,
    juiz_nli_llm,
    kappa_fleiss,
    voto,
)

familia = os.environ.get("FAMILIA", "transformer")
nome_tarefa = os.environ.get("TAREFA", "polaridade")
tarefa = tarefas[nome_tarefa]
rotulos = tarefa["rotulos"]
modelos = dict(familias[familia])
if familia == "deepseek":
    modelos["llm"] = os.environ.get("LLM", modelos["llm"])
nome_familia = modelos["llm"].split("/")[-1].lower() if familia == "deepseek" else familia
pasta = f"saida/{versao}"

random.seed(semente)
torch.manual_seed(semente)

lexico = carregar_lexico(tarefa["lexico"], rotulos) if os.path.exists(tarefa["lexico"]) else {}
if not lexico:
    print(f"Léxico não encontrado em {tarefa['lexico']}, o juiz de atenção vai se abster em todas as frases.\n")

decoder = familia == "deepseek"
tokenizer, model = carregar(modelos["llm"] if decoder else modelos["encoder"], decoder=decoder)
os.makedirs(pasta, exist_ok=True)

for conjunto in conjuntos + ["calibracao"]:
    arquivo = f"dados/calibracao_{nome_tarefa}.csv" if conjunto == "calibracao" else f"dados/amostra_{conjunto}.csv"
    with open(arquivo, encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))
    textos = [linha["texto"] for linha in linhas]
    tem_gabarito = conjunto in (tarefa["gabarito"], "calibracao")

    if decoder:
        scores = {
            "embeddings": juiz_embeddings(tokenizer, model, textos, tarefa["prototipos"], tarefa["resumo"]),
            "nli": juiz_nli_llm(tokenizer, model, textos, rotulos, tarefa["pergunta"]),
            "atencao": juiz_atencao(tokenizer, model, textos, lexico, rotulos, decoder=True),
        }
    else:
        scores = {
            "embeddings": juiz_embeddings(tokenizer, model, textos, tarefa["prototipos"]),
            "nli": juiz_nli(modelos["nli"], textos, rotulos, tarefa["hipotese"]),
            "atencao": juiz_atencao(tokenizer, model, textos, lexico, rotulos),
        }

    juizes = list(scores)
    controle = juiz_lexico(textos, lexico, rotulos)
    votos = [[voto(scores[juiz][i], rotulos) for juiz in juizes] for i in range(len(textos))]
    saida = f"{pasta}/juizes_{nome_familia}_{nome_tarefa}_{conjunto}.csv"

    concordancias = []
    maiorias = []
    with open(saida, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        cabecalho = ["id", "texto"]
        for juiz in juizes:
            cabecalho += [f"{juiz}_{rotulo}" for rotulo in rotulos] + [f"{juiz}_voto"]
        cabecalho += [f"lexico_{rotulo}" for rotulo in rotulos] + ["lexico_voto"]
        cabecalho += ["maioria", "concordancia", "gabarito"]
        writer.writerow(cabecalho)

        for i, linha in enumerate(linhas):
            validos = [v for v in votos[i] if v]
            maioria = max(validos, key=validos.count) if validos else ""
            concordancia = validos.count(maioria) if validos else 0
            concordancias.append(concordancia)
            maiorias.append(maioria)

            registro = [linha["id"], linha["texto"]]
            for j, juiz in enumerate(juizes):
                registro += [f"{s:.4f}" for s in scores[juiz][i]] + [votos[i][j] or ""]
            registro += [f"{s:.4f}" for s in controle[i]] + [voto(controle[i], rotulos) or ""]
            registro += [maioria, concordancia, linha["gabarito"] if tem_gabarito else ""]
            writer.writerow(registro)

    print(f"--- {nome_tarefa.upper()} | {nome_familia.upper()} | {len(textos)} FRASES DO {conjunto.upper()} ---")
    for j, juiz in enumerate(juizes):
        contagem = {rotulo: 0 for rotulo in rotulos + [None]}
        for v in votos:
            contagem[v[j]] += 1
        print(f"{juiz}: " + " | ".join(f"{r or 'absteve'}: {c}" for r, c in contagem.items()))
    for n in range(len(juizes), -1, -1):
        print(f"{n} juízes concordam: {concordancias.count(n)} frases")
    kappa, usadas = kappa_fleiss(votos, rotulos)
    print(f"Kappa de Fleiss: {kappa:.4f} (nas {usadas} frases em que nenhum juiz se absteve)")

    if tem_gabarito:
        gabarito = [linha["gabarito"] for linha in linhas]
        for j, juiz in enumerate(juizes):
            acertos = sum(v[j] == g for v, g in zip(votos, gabarito))
            print(f"Acurácia do juiz {juiz}: {acertos / len(gabarito) * 100:.2f}%")
        acertos = sum(m == g for m, g in zip(maiorias, gabarito))
        print(f"Acurácia da maioria: {acertos / len(gabarito) * 100:.2f}% (chance: {100 / len(rotulos):.2f}%)")
    print(f"Resultado frase a frase em {saida}\n")
