import csv
import glob
import os
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

limites = [-0.6, -0.2, 0.2, 0.6]


def r2(x, y):
    a, b = np.polyfit(x, y, 1)
    residuo = y - (a * x + b)
    return 1 - (residuo ** 2).sum() / ((y - y.mean()) ** 2).sum()


for arquivo in sorted(glob.glob("saida/polaridade*.csv")):
    with open(arquivo, encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))

    modelos = [coluna[:-6] for coluna in linhas[0] if coluna.endswith("_score")]
    tem_rotulo = "rotulo" in linhas[0]
    base = os.path.splitext(os.path.basename(arquivo))[0]

    figura, eixos = plt.subplots(len(modelos), 2, figsize=(11, 3.4 * len(modelos)), squeeze=False)
    print(f"--- CURVAS EM {arquivo} ({len(linhas)} frases) ---\n")

    for linha_eixo, modelo in zip(eixos, modelos):
        scores = np.sort(np.array([float(linha[f"{modelo}_score"]) for linha in linhas]))
        p = (np.arange(len(scores)) + 0.5) / len(scores)
        logit = np.log(p / (1 - p))

        r2_reta = r2(p, scores)
        r2_s = r2(logit, scores)
        fatias = np.histogram(scores, bins=[-1.0001] + limites + [1.0001])[0] / len(scores) * 100
        meio, pontas = fatias[2], (fatias[0] + fatias[4]) / 2
        intermediarias = (fatias[1] + fatias[3]) / 2
        if all(abs(f - 20) < 5 for f in fatias):
            formato = "reta, as cinco faixas com perto de 20%"
        elif intermediarias < 10 and meio > 20 and pontas > 20:
            formato = "em degraus, cai quase tudo em três níveis e as faixas intermediárias somem, ou seja, age como ternário"
        elif meio > pontas:
            formato = "curva em S, concentra no meio e aperta as pontas"
        else:
            formato = "em U, concentra nas pontas e esvazia as faixas do meio"
        print(f"{modelo}: R² da reta {r2_reta:.3f} | R² da curva em S {r2_s:.3f}")
        print("  % por faixa (uma reta daria 20% em cada): " + " | ".join(f"{f:.1f}" for f in fatias))
        print(f"  formato: {formato}")

        histograma, quantis = linha_eixo
        histograma.hist(scores, bins=40, color="#2f6b5a")
        for limite in limites:
            histograma.axvline(limite, color="#8a5a00", linestyle="--", linewidth=1)
        histograma.set_title(f"{modelo}: distribuição do score e faixas de 0,4")
        histograma.set_xlim(-1, 1)

        quantis.plot(p, scores, color="#2f6b5a", label="score ordenado")
        quantis.plot(p, np.polyval(np.polyfit(p, scores, 1), p), "--", color="#5c6862", label=f"reta (R² {r2_reta:.2f})")
        quantis.plot(p, np.polyval(np.polyfit(logit, scores, 1), logit), ":", color="#3f4fa3", label=f"curva em S (R² {r2_s:.2f})")
        quantis.set_title(f"{modelo}: score por posição na fila")
        quantis.set_ylim(-1.05, 1.05)
        quantis.legend(fontsize=8)

        if tem_rotulo:
            for classe in ["negativo", "neutro", "positivo"]:
                da_classe = np.array([float(l[f"{modelo}_score"]) for l in linhas if l["rotulo"] == classe])
                faixa_media = np.percentile(da_classe, [25, 50, 75])
                print(f"  {classe}: mediana {faixa_media[1]:+.3f}, metade do meio entre {faixa_media[0]:+.3f} e {faixa_media[2]:+.3f}")

    figura.tight_layout()
    imagem = f"saida/curvas_{base}.png"
    figura.savefig(imagem, dpi=120)
    print(f"Gráfico salvo em {imagem}\n")
