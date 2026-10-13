import csv
import os
import numpy as np
from protocolo.config import conjuntos, familias, reamostragens, semente, tarefas, versao

juizes = ["embeddings", "nli", "atencao"]
pasta = f"saida/{versao}"
llm = os.environ.get("LLM", familias["deepseek"]["llm"]).split("/")[-1].lower()
rng = np.random.default_rng(semente)


def ler(nome_familia, tarefa, conjunto):
    with open(f"{pasta}/juizes_{nome_familia}_{tarefa}_{conjunto}.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def kappa_cohen(a, b):
    if len(a) == 0:
        return np.nan
    rotulos = sorted(set(a) | set(b))
    observada = np.mean(a == b)
    esperada = sum(np.mean(a == r) * np.mean(b == r) for r in rotulos)
    return 1.0 if esperada == 1 else (observada - esperada) / (1 - esperada)


def intervalo(medida, n):
    valores = [medida(rng.integers(0, n, n)) for _ in range(reamostragens)]
    return np.nanpercentile(valores, 2.5), np.nanpercentile(valores, 97.5)


def media_onde(acerto, filtro):
    return lambda indices: acerto[indices][filtro[indices]].mean() if filtro[indices].any() else np.nan


resultados = []


def registrar(tarefa, conjunto, medida, funcao, n, frases):
    valor = funcao(np.arange(n))
    inferior, superior = intervalo(funcao, n)
    resultados.append([tarefa, conjunto, medida, frases, f"{valor:.4f}", f"{inferior:.4f}", f"{superior:.4f}"])
    print(f"  {medida:<48} {valor * 100:6.1f}%  [{inferior * 100:5.1f}, {superior * 100:5.1f}]  n={frases}")


for tarefa in tarefas:
    for conjunto in conjuntos:
        t = ler("transformer", tarefa, conjunto)
        d = ler(llm, tarefa, conjunto)
        if [linha["id"] for linha in t] != [linha["id"] for linha in d]:
            raise SystemExit(f"As duas famílias não usaram as mesmas frases em {tarefa}/{conjunto}")
        n = len(t)
        print(f"--- {tarefa.upper()} NO {conjunto.upper()}: TRANSFORMER X {llm.upper()} ---")

        for juiz in juizes + ["maioria"]:
            coluna = juiz if juiz == "maioria" else f"{juiz}_voto"
            a = np.array([linha[coluna] for linha in t])
            b = np.array([linha[coluna] for linha in d])
            igual = (a == b) & (a != "")
            ambos = (a != "") & (b != "")
            registrar(tarefa, conjunto, f"mesmo rótulo nas duas famílias ({juiz})", lambda i: igual[i].mean(), n, n)
            registrar(
                tarefa, conjunto, f"kappa de Cohen ({juiz})",
                lambda i: kappa_cohen(a[i][ambos[i]], b[i][ambos[i]]), n, int(ambos.sum()),
            )

        maioria_t = np.array([linha["maioria"] for linha in t])
        maioria_d = np.array([linha["maioria"] for linha in d])
        convergem = (maioria_t == maioria_d) & (maioria_t != "")
        nivel = np.minimum(
            np.array([int(linha["concordancia"]) for linha in t]),
            np.array([int(linha["concordancia"]) for linha in d]),
        )
        for minimo in [3, 2, 1]:
            filtro = nivel == minimo
            registrar(
                tarefa, conjunto, f"famílias convergem quando as duas têm {minimo}/3",
                media_onde(convergem, filtro), n, int(filtro.sum()),
            )

        gabarito = np.array([linha["gabarito"] for linha in t])
        if (gabarito != "").all():
            registrar(tarefa, conjunto, "acerto do transformer", lambda i: (maioria_t[i] == gabarito[i]).mean(), n, n)
            registrar(tarefa, conjunto, f"acerto do {llm}", lambda i: (maioria_d[i] == gabarito[i]).mean(), n, n)
            acerto = maioria_t == gabarito
            registrar(tarefa, conjunto, "acerto onde as famílias convergem", media_onde(acerto, convergem), n, int(convergem.sum()))
            registrar(tarefa, conjunto, "acerto do transformer onde as famílias divergem", media_onde(acerto, ~convergem), n, int((~convergem).sum()))
        print()

with open(f"{pasta}/convergencia_{llm}.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["tarefa", "conjunto", "medida", "frases", "valor", "ic95_inferior", "ic95_superior"])
    writer.writerows(resultados)
print(f"Tabela com intervalo de confiança de 95% ({reamostragens} reamostragens) em {pasta}/convergencia_{llm}.csv")
