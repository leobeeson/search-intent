import os
import torch
import evaluate
import optuna
import pickle
import numpy as np


from datasets import Dataset, Features, Value, ClassLabel, DatasetDict, load_from_disk
from transformers import DataCollatorWithPadding
from transformers import Trainer, TrainingArguments
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler


def perform_hyperparameter_optimization():
    print(torch.cuda.is_available())
    
    
    results_dir = os.path.join(".temp", "results")
    logs_dir = os.path.join(".temp", "logs")
    model_dir = os.path.join("ml_models", "final_model")
    optuna_study_dir = os.path.join(".temp", "optuna_study")
    optuna_pickles_dir = os.path.join(".temp", "optuna_pickles")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    checkpoint = "distilbert-base-uncased"

    raw_datasets = load_from_disk(os.environ['PATH_ARROW_CACHE'])

    tokenizer = DistilBertTokenizer.from_pretrained(checkpoint)

    def tokenize_function(example):
        return tokenizer(example["text"])

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

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

    max_f1_score = 0

    def objective(trial):
        global max_f1_score

        # Define hyperparameter search space
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
        num_train_epochs = trial.suggest_int("num_train_epochs", 1, 3)
        per_device_train_batch_size = trial.suggest_categorical("per_device_train_batch_size", [8, 16])
        warmup_steps = trial.suggest_int("warmup_steps", 0, 1000, step=100)
        weight_decay = trial.suggest_float("weight_decay", 0, 0.1, step=0.01)

        # Update training arguments with hyperparameter values
        training_args.learning_rate = learning_rate
        training_args.num_train_epochs = num_train_epochs
        training_args.per_device_train_batch_size = per_device_train_batch_size
        training_args.warmup_steps = warmup_steps
        training_args.weight_decay = weight_decay

        # Instantiate a fresh model for this trial
        model = DistilBertForSequenceClassification.from_pretrained(checkpoint, num_labels=1419)

        # Train the model
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

        # Evaluate the model
        metrics = trainer.evaluate()

        # If the current trial's f1-score is greater than the max_f1_score, save the model and update max_f1_score
        if metrics["eval_f1"] > max_f1_score or trial.number == 0:
            max_f1_score = metrics["eval_f1"]
            model.save_pretrained(model_dir)

        # Return the metric we want to optimize
        return metrics["eval_f1"]

    tokenizer.save_pretrained(model_dir)

    storage_name = f"sqlite:////{optuna_study_dir}/my_study.db"

    pruner = optuna.pruners.MedianPruner()

    study = optuna.create_study(direction="maximize", 
                            storage=storage_name, 
                            study_name='my_study', 
                            load_if_exists=True,
                            pruner=pruner,
                            sampler=optuna.samplers.TPESampler())
    

    num_trials_per_round = 5
    num_rounds = 4  # 4*5=20 trials

    for i in range(num_rounds):
        # run optuna.optimize for num_trials_per_round
        study.optimize(objective, n_trials=num_trials_per_round, timeout=36000)
        # save the study object
        with open(f"{optuna_pickles_dir}/my_study.pkl", 'wb') as f:
            pickle.dump(study, f)
