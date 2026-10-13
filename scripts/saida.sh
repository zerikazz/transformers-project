#!/usr/bin/env bash
set -euo pipefail

CONFIGURACAO_GCLOUD="${GCLOUD_CONFIG:-transformers}"
PASTA_NO_DRIVE="${DRIVE_FOLDER:-1HmLUc66KUKbCVYeefK1lS5D25RPVtlFX}"
NOME_NO_DRIVE="${DRIVE_FILE:-saida.tar.gz}"
DRIVE="https://www.googleapis.com/drive/v3/files"
DRIVE_UPLOAD="https://www.googleapis.com/upload/drive/v3/files"

ok()    { printf '  . %s\n' "$1"; }
aviso() { printf '  ! %s\n' "$1" >&2; }

if gcloud config configurations describe "$CONFIGURACAO_GCLOUD" >/dev/null 2>&1; then
  export CLOUDSDK_ACTIVE_CONFIG_NAME="$CONFIGURACAO_GCLOUD"
fi

TOKEN="$(gcloud auth print-access-token 2>/dev/null)" || {
  aviso "Sem login no Google. Rode: CLOUDSDK_ACTIVE_CONFIG_NAME=$CONFIGURACAO_GCLOUD gcloud auth login --enable-gdrive-access"
  exit 1
}

drive() {
  curl -sS --fail-with-body -H "Authorization: Bearer $TOKEN" "$@"
}

id_do_arquivo() {
  drive -G "$DRIVE" \
    --data-urlencode "q=name = '$NOME_NO_DRIVE' and '$PASTA_NO_DRIVE' in parents and trashed = false" \
    --data-urlencode "fields=files(id)" \
    | python -c "import json, sys; arquivos = json.load(sys.stdin).get('files') or []; print(arquivos[0]['id'] if arquivos else '', end='')"
}

comando_push() {
  [[ -d saida ]] || { aviso "Não tem pasta saida/ para subir"; exit 1; }
  local pacote id metodo url
  pacote="$(mktemp).tar.gz"
  tar -czf "$pacote" saida
  id="$(id_do_arquivo)"
  if [[ -n "$id" ]]; then
    metodo="PATCH"; url="$DRIVE_UPLOAD/$id?uploadType=resumable"
    metadados="{}"
  else
    metodo="POST"; url="$DRIVE_UPLOAD?uploadType=resumable"
    metadados="{\"name\": \"$NOME_NO_DRIVE\", \"parents\": [\"$PASTA_NO_DRIVE\"]}"
  fi
  local sessao
  sessao="$(drive -X "$metodo" "$url" -D - -o /dev/null \
    -H "Content-Type: application/json; charset=UTF-8" \
    -H "X-Upload-Content-Type: application/gzip" \
    -d "$metadados" | tr -d '\r' | sed -n 's/^[Ll]ocation: //p')"
  [[ -n "$sessao" ]] || { aviso "O Drive não abriu o envio"; rm -f "$pacote"; exit 1; }
  drive -X PUT "$sessao" -H "Content-Type: application/gzip" --data-binary @"$pacote" -o /dev/null
  ok "$NOME_NO_DRIVE ($(du -h "$pacote" | cut -f1)) atualizado no Drive"
  rm -f "$pacote"
}

comando_pull() {
  local id pacote
  id="$(id_do_arquivo)"
  [[ -n "$id" ]] || { aviso "Não achei '$NOME_NO_DRIVE' na pasta do Drive. Suba com: scripts/saida.sh push"; exit 1; }
  pacote="$(mktemp).tar.gz"
  drive -L "$DRIVE/$id?alt=media" -o "$pacote"
  tar -xzf "$pacote"
  rm -f "$pacote"
  ok "saida/ baixada do Drive"
}

case "${1:-}" in
  push) comando_push ;;
  pull) comando_pull ;;
  *)
    echo "Uso: scripts/saida.sh push | pull"
    echo "  push  empacota a saida/ (resultados e modelos treinados) e sobe como $NOME_NO_DRIVE na pasta do Drive"
    echo "  pull  baixa o $NOME_NO_DRIVE da pasta do Drive e abre na saida/"
    exit 1
    ;;
esac
