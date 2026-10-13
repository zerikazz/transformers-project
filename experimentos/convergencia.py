import csv
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from protocolo.config import conjuntos, familias, reamostragens, semente, tarefas, versao

juizes = ["embeddings", "nli", "atencao"]
pasta = f"saida/{versao}"
llm = os.environ.get("LLM", familias["deepseek"]["llm"]).split("/")[-1].lower()
rng = np.random.default_rng(semente)


def ler(nome_familia, tarefa, conjunto):
    with open(f"{pasta}/juizes_{nome_familia}_{tarefa}_{conjunto}.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def colunas_dos_juizes(linhas, rotulos):
    return np.array([[float(linha[f"{juiz}_{rotulo}"]) for juiz in juizes for rotulo in rotulos] for linha in linhas])


def pesos(nome_familia, tarefa, rotulos):
    calibracao = ler(nome_familia, tarefa, "calibracao")
    modelo = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=semente))
    return modelo.fit(colunas_dos_juizes(calibracao, rotulos), [linha["gabarito"] for linha in calibracao])


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
    rotulos = tarefas[tarefa]["rotulos"]
    pesos_t = pesos("transformer", tarefa, rotulos)
    pesos_d = pesos(llm, tarefa, rotulos)
    for conjunto in conjuntos:
        t = ler("transformer", tarefa, conjunto)
        d = ler(llm, tarefa, conjunto)
        if [linha["id"] for linha in t] != [linha["id"] for linha in d]:
            raise SystemExit(f"As duas famílias não usaram as mesmas frases em {tarefa}/{conjunto}")
        n = len(t)
        for linhas, modelo in [(t, pesos_t), (d, pesos_d)]:
            for linha, previsto in zip(linhas, modelo.predict(colunas_dos_juizes(linhas, rotulos))):
                linha["ponderado"] = previsto
        print(f"--- {tarefa.upper()} NO {conjunto.upper()}: TRANSFORMER X {llm.upper()} ---")

        for juiz in juizes + ["lexico", "maioria", "ponderado"]:
            coluna = juiz if juiz in ("maioria", "ponderado") else f"{juiz}_voto"
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

        for nome, linhas in [("transformer", t), (llm, d)]:
            atencao = np.array([linha["atencao_voto"] for linha in linhas])
            lexico = np.array([linha["lexico_voto"] for linha in linhas])
            registrar(tarefa, conjunto, f"atenção vota igual ao léxico sozinho ({nome})", lambda i: (atencao[i] == lexico[i]).mean(), n, n)

        gabarito = np.array([linha["gabarito"] for linha in t])
        if (gabarito != "").all():
            for nome, linhas in [("transformer", t), (llm, d)]:
                for coluna in ["atencao_voto", "lexico_voto", "maioria", "ponderado"]:
                    decisao = np.array([linha[coluna] for linha in linhas])
                    registrar(tarefa, conjunto, f"acerto {coluna} ({nome})", lambda i: (decisao[i] == gabarito[i]).mean(), n, n)
            acerto = maioria_t == gabarito
            registrar(tarefa, conjunto, "acerto onde as famílias convergem", media_onde(acerto, convergem), n, int(convergem.sum()))
            registrar(tarefa, conjunto, "acerto do transformer onde as famílias divergem", media_onde(acerto, ~convergem), n, int((~convergem).sum()))
        print()

with open(f"{pasta}/convergencia_{llm}.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["tarefa", "conjunto", "medida", "frases", "valor", "ic95_inferior", "ic95_superior"])
    writer.writerows(resultados)
print(f"Tabela com intervalo de confiança de 95% ({reamostragens} reamostragens) em {pasta}/convergencia_{llm}.csv")
