import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. Definir o modelo RoBERTa pré-treinado em Português e o mapeamento de classes
model_name_roberta = "cardiffnlp/xlm-roberta-base-tweet-sentiment-pt"

# Mapeamento para 3 classes de sentimento
id2label = {0: "Negativo", 1: "Neutro", 2: "Positivo"}

# 2. Carregar o tokenizador e o modelo
tokenizer = AutoTokenizer.from_pretrained(model_name_roberta)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name_roberta, 
    num_labels=len(id2label)
)

# Colocar o modelo em modo de avaliação (desativa dropout)
model.eval()

# 3. Frases de teste
phrases = [
    "Este produto superou minhas expectativas, recomendo muito!",
    "Achei a entrega demorada e o produto veio danificado.",
    "O pedido chegou dentro do prazo normal."
]

# 4. Tokenização das frases (O RoBERTa insere <s> no início e </s> no final)
inputs = tokenizer(
    phrases,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors="pt"
)

# 5. Classificação de Sentimento (Inferência sem gradientes)
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# 6. Converter logits em probabilidades (Softmax)
probabilities = F.softmax(logits, dim=-1)

# 7. Obter a classe com maior probabilidade
predicted_classes = torch.argmax(probabilities, dim=-1)

# 8. Exibir os resultados organizados
print("--- RESULTADOS DA CLASSIFICAÇÃO COM ROBERTA-PT ---\n")
for i, phrase in enumerate(phrases):
    pred_id = predicted_classes[i].item()
    pred_label = id2label[pred_id]
    confidence = probabilities[i][pred_id].item() * 100
    
    print(f"Frase: \"{phrase}\"")
    print(f"Predição: {pred_label} (Confiança: {confidence:.2f}%)")
    print(f"Probabilidades -> Negativo: {probabilities[i][0]:.2f} | Neutro: {probabilities[i][1]:.2f} | Positivo: {probabilities[i][2]:.2f}")
    print("-" * 60)