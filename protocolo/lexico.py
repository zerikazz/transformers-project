import csv
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from protocolo.juizes import emocoes

entrada = "dados/goemotions_treino.csv"
saida = os.environ.get("LEXICO", "dados/lexico.csv")
palavras_por_emocao = int(os.environ.get("PALAVRAS", 100))
frequencia_minima = 5


def entropia(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


with open(entrada, encoding="utf-8") as f:
    linhas = list(csv.DictReader(f))

vetorizador = CountVectorizer(
    binary=True,
    lowercase=True,
    strip_accents="unicode",
    token_pattern=r"(?u)\b[^\W\d_]{2,}\b",
    min_df=frequencia_minima,
)
presenca = vetorizador.fit_transform(linha["texto"] for linha in linhas).tocsc()
vocabulario = vetorizador.get_feature_names_out()
total = presenca.shape[0]
com_palavra = np.asarray(presenca.sum(axis=0)).ravel()
sem_palavra = total - com_palavra

os.makedirs(os.path.dirname(saida) or ".", exist_ok=True)

with open(saida, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["palavra", "emocao", "ganho_de_informacao"])

    for emocao in emocoes:
        alvo = np.array([linha["rotulo"] == emocao for linha in linhas])
        p_emocao = alvo.mean()
        emocao_com_palavra = np.asarray(presenca[alvo].sum(axis=0)).ravel()
        emocao_sem_palavra = alvo.sum() - emocao_com_palavra

        p_dado_palavra = emocao_com_palavra / com_palavra
        p_dado_ausencia = emocao_sem_palavra / np.maximum(sem_palavra, 1)
        ganho = entropia(p_emocao) - (
            com_palavra / total * entropia(p_dado_palavra)
            + sem_palavra / total * entropia(p_dado_ausencia)
        )
        ganho[p_dado_palavra <= p_emocao] = 0

        melhores = np.argsort(-ganho)[:palavras_por_emocao]
        for i in melhores:
            if ganho[i] > 0:
                writer.writerow([vocabulario[i], emocao, f"{ganho[i]:.6f}"])

        print(f"{emocao}: " + ", ".join(vocabulario[i] for i in melhores[:12]))

print(f"Léxico com até {palavras_por_emocao} palavras por emoção salvo em {saida}")
