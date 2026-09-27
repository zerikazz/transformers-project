import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name_bert = "pysentimiento/bertweet-pt-sentiment"

id2label = {0: "Negativo", 1: "Neutro", 2: "Positivo"}

tokenizer = AutoTokenizer.from_pretrained(model_name_bert)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name_bert, 
    num_labels=len(id2label)
)

model.eval()

phrases = [
    "Este produto superou minhas expectativas, recomendo muito!",
    "Achei a entrega demorada e o produto veio danificado.",
    "O pedido chegou dentro do prazo normal."
]

inputs = tokenizer(
    phrases,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors="pt"
)

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    
probabilities = F.softmax(logits, dim=-1)

predicted_classes = torch.argmax(probabilities, dim=-1)

print("--- RESULTADOS DA CLASSIFICAÇÃO COM BERTIMBAU ---\n")
for i, phrase in enumerate(phrases):
    pred_id = predicted_classes[i].item()
    pred_label = id2label[pred_id]
    confidence = probabilities[i][pred_id].item() * 100
    
    print(f"Frase: \"{phrase}\"")
    print(f"Predição: {pred_label} (Confiança: {confidence:.2f}%)")
    print(f"Probabilidades -> Negativo: {probabilities[i][0]:.2f} | Neutro: {probabilities[i][1]:.2f} | Positivo: {probabilities[i][2]:.2f}")
    print("-" * 60)