import csv
import gzip
import os
import xml.etree.ElementTree as ET
from huggingface_hub import hf_hub_download

repo_carolina = "carolina-c4ai/corpus-carolina"
arquivos_soc = ["corpus/social_media/SOCa.xml.gz", "corpus/social_media/SOCb.xml.gz"]
tei = "{http://www.tei-c.org/ns/1.0}"
min_palavras = 3
saida = "dados/carolina_soc.csv"

os.makedirs("dados", exist_ok=True)

vistos = set()
total = 0

with open(saida, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "documento", "texto"])

    for arquivo in arquivos_soc:
        caminho = hf_hub_download(repo_carolina, arquivo, repo_type="dataset")

        with gzip.open(caminho, "rb") as gz:
            for _, elemento in ET.iterparse(gz):
                if elemento.tag != f"{tei}TEI":
                    continue

                documento = elemento.find(f"{tei}teiHeader/{tei}fileDesc/{tei}titleStmt/{tei}title").text

                for paragrafo in elemento.findall(f".//{tei}body/{tei}p"):
                    texto = " ".join((paragrafo.text or "").split())
                    if len(texto.split()) < min_palavras or texto in vistos:
                        continue
                    vistos.add(texto)
                    writer.writerow([total, documento, texto])
                    total += 1

                elemento.clear()

print(f"{total} frases do Carolina (soc) salvas em {saida}")
