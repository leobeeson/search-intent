import os
import torch
import evaluate
import numpy as np


from datasets import load_from_disk
from transformers import DataCollatorWithPadding
from transformers import Trainer, TrainingArguments
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification


# Make sure to run `!pip install --upgrade accelerate`


def train_distilbert_model():
    print(torch.cuda.is_available())
    
    
    results_dir = os.path.join(".temp", "results")
    logs_dir = os.path.join(".temp", "logs")
    model_dir = os.path.join("ml_models", "final_model")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    checkpoint = "distilbert-base-uncased"

    raw_datasets = load_from_disk(os.environ['PATH_ARROW_CACHE'])
    model = DistilBertForSequenceClassification.from_pretrained(checkpoint, num_labels=1419)
    tokenizer = DistilBertTokenizer.from_pretrained(checkpoint)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def tokenize_function(example):
        return tokenizer(example["text"])
    
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

    
    metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels, average="macro")


    training_args = TrainingArguments(
            output_dir=results_dir,
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            learning_rate=2e-5,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=logs_dir,
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            seed=42,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            gradient_accumulation_steps=1,
            push_to_hub=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )

    trainer.train()

    output_dir=model_dir
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

