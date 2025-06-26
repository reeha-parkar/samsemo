# Fine-tuning Wav2Vec2.0 on SAMSEMO Audio for Emotion Recognition

!pip install datasets transformers evaluate torchaudio -q

import os
import numpy as np
from datasets import load_dataset, Audio, ClassLabel
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForSequenceClassification,
    TrainingArguments,
    Trainer,
    DefaultDataCollator
)
import evaluate

# 1. Load SAMSEMO (English split) and map emotion labels to ClassLabel
dataset = load_dataset("SamsungNLP/SAMSEMO", "en")
    
# Combine train/validation split if not provided; otherwise use built-in splits
if "train" not in dataset:
    dataset = dataset["train"].train_test_split(test_size=0.1)
train_ds = dataset["train"]
eval_ds  = dataset["test"] if "test" in dataset else dataset["test"]

# 2. Cast the audio column to uniform sampling rate
train_ds = train_ds.cast_column("audio", Audio(sampling_rate=16000))
eval_ds  = eval_ds.cast_column ("audio", Audio(sampling_rate=16000))

# 3. Create a ClassLabel feature for emotions
emotions = sorted(set(train_ds["emotion"]))
label2id = {label: idx for idx, label in enumerate(emotions)}
id2label = {idx: label for label, idx in label2id.items()}
num_labels = len(emotions)

# 4. Load pretrained processor & model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-base",
    num_labels=num_labels,
    label2id=label2id,
    id2label=id2label,
    problem_type="single_label_classification"
)

# 5. Preprocessing function: extract input_values and labels
def preprocess(batch):
    # batch["audio"]["array"] is a numpy array
    audio = batch["audio"]["array"]
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    batch["input_values"] = inputs.input_values[0]
    batch["attention_mask"] = inputs.attention_mask[0]
    batch["labels"] = label2id[batch["emotion"]]
    return batch

train_ds = train_ds.map(preprocess, remove_columns=train_ds.column_names)
eval_ds  = eval_ds.map (preprocess, remove_columns=eval_ds.column_names)

# 6. Data collator
data_collator = DefaultDataCollator()

# 7. Metrics
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(pred):
    logits = pred.predictions
    preds = np.argmax(logits, axis=-1)
    labels = pred.label_ids
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "f1": f1.compute(predictions=preds, references=labels, average="weighted")["f1"]
    }

# 8. Training arguments
training_args = TrainingArguments(
    output_dir="wav2vec2-samsemo",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    evaluation_strategy="epoch",
    num_train_epochs=5,
    learning_rate=1e-4,
    save_strategy="epoch",
    logging_dir="logs",
    logging_steps=50,
    warmup_steps=500
)

# 9. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=processor.feature_extractor,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# 10. Train & Evaluate
trainer.train()
trainer.evaluate()

