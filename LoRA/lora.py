import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftModel
)
import numpy as np
from sklearn.metrics import accuracy_score

# Configurações
MODEL_NAME = "google/mobilebert-uncased"  # Um modelo pequeno para demonstração
OUTPUT_DIR = "./results-lora-modernbert"
NUM_LABELS = 2  # Para classificação binária
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Carregar dataset (usando SST-2 como exemplo para classificação de sentimento)
dataset = load_dataset("glue", "sst2")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Função para tokenizar os exemplos
def tokenize_function(examples):
    return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=128)

# Tokenizar o dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Configurar o modelo base
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=NUM_LABELS,
)

# Configurar LoRA
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,  # Classificação de sequência
    r=8,  # Rank da matriz LoRA
    lora_alpha=16,  # Parâmetro de escala
    lora_dropout=0.1,  # Dropout para regularização
    bias="none",  # Não ajustar vieses
    # Lista de módulos para ajustar - depende da arquitetura do modelo
    target_modules=["query", "key", "value"],
)

# Aplicar configuração LoRA ao modelo
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()  # Mostrar a redução de parâmetros treináveis

# Configurar métricas de avaliação
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

# Configurar treinamento
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=1e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
)

# Configurar o Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)

# Treinar o modelo
trainer.train()

# Salvar o modelo ajustado
model.save_pretrained(f"{OUTPUT_DIR}/final_model")

# Carregar e usar o modelo ajustado
def load_finetuned_model():
    # Carregar o modelo base
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
    )
    
    # Carregar os adaptadores LoRA
    model = PeftModel.from_pretrained(base_model, f"{OUTPUT_DIR}/final_model")
    
    return model, tokenizer

# Função para fazer previsões com o modelo ajustado
def predict(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    pred_class = torch.argmax(probabilities, dim=-1).item()
    
    return {"class": pred_class, "probability": probabilities[0][pred_class].item()}

# Exemplo de uso
finetuned_model, tokenizer = load_finetuned_model()
sample_text = "This movie was absolutely fantastic!"
prediction = predict(sample_text, finetuned_model, tokenizer)
print(f"Prediction: Class {prediction['class']} with probability {prediction['probability']:.4f}")