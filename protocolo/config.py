versao = "v1"
semente = 42
amostra = 300
reamostragens = 1000

conjuntos = ["carolina", "tweets", "goemotions"]

familias = {
    "transformer": {
        "encoder": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "nli": "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
    },
    "deepseek": {
        "llm": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    },
}

tarefas = {
    "polaridade": {
        "rotulos": ["negativo", "neutro", "positivo"],
        "prototipos": {
            "negativo": ["Que horror, odiei isso.", "Isso foi péssimo, uma decepção.", "Estou muito chateado com isso."],
            "neutro": ["O jogo começa às oito horas.", "A loja fica na rua principal.", "Hoje é terça-feira."],
            "positivo": ["Adorei, foi maravilhoso!", "Que notícia boa, estou feliz.", "Isso ficou ótimo, parabéns."],
        },
        "hipotese": "O sentimento deste texto é {}.",
        "pergunta": "Texto: \"{texto}\"\nPergunta: O sentimento deste texto é {rotulo}? Responda apenas sim ou não.\nResposta:",
        "resumo": "Texto: \"{texto}\"\nEm uma palavra, o sentimento deste texto é:",
        "lexico": "dados/lexico_polaridade.csv",
        "treino": "dados/tweets_train.csv",
        "gabarito": "tweets",
    },
    "emocao": {
        "rotulos": ["alegria", "tristeza", "raiva", "medo", "surpresa", "nojo"],
        "prototipos": {
            "alegria": ["Estou muito feliz hoje!", "Que notícia maravilhosa, adorei!", "Isso me deixou radiante."],
            "tristeza": ["Estou muito triste.", "Que saudade, meu coração está partido.", "Isso me deixou para baixo."],
            "raiva": ["Estou com muita raiva!", "Que ódio, isso é um absurdo!", "Fiquei furioso com isso."],
            "medo": ["Estou com muito medo.", "Isso me deixa apavorado.", "Tenho pavor do que pode acontecer."],
            "surpresa": ["Nossa, não acredito!", "Que surpresa, eu não esperava por isso!", "Fiquei chocado com a novidade."],
            "nojo": ["Que nojo!", "Isso é repugnante.", "Senti nojo daquilo."],
        },
        "hipotese": "Este texto expressa {}.",
        "pergunta": "Texto: \"{texto}\"\nPergunta: Este texto expressa {rotulo}? Responda apenas sim ou não.\nResposta:",
        "resumo": "Texto: \"{texto}\"\nEm uma palavra, a emoção deste texto é:",
        "lexico": "dados/lexico.csv",
        "treino": "dados/goemotions_treino.csv",
        "gabarito": "goemotions",
    },
}

neutros_llm = ["N/A", "", "[sem texto]"]
