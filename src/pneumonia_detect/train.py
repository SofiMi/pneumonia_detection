import logging
import subprocess
from pathlib import Path
from typing import Dict, List

import evaluate
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import torch
from datasets import ClassLabel, Dataset, Image
from imblearn.over_sampling import RandomOverSampler
from omegaconf import DictConfig, OmegaConf
from PIL import ImageFile
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomAdjustSharpness,
    RandomRotation,
    Resize,
    ToTensor,
)
from transformers import (
    Trainer,
    TrainingArguments,
    ViTForImageClassification,
    ViTImageProcessor,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

log = logging.getLogger(__name__)


def get_git_commit_id() -> str:
    try:
        commit_hash = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .strip()
            .decode("utf-8")
        )
        return commit_hash
    except Exception:
        log.warning("Не удалось получить Git commit ID. Используем 'unknown'.")
        return "unknown"


def save_static_plots(log_history: List[Dict], output_dir: str):
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    train_loss = []
    eval_loss = []
    eval_acc = []

    for entry in log_history:
        if "loss" in entry:
            train_loss.append((entry["step"], entry["loss"]))
        if "eval_loss" in entry:
            eval_loss.append((entry["step"], entry["eval_loss"]))
        if "eval_accuracy" in entry:
            eval_acc.append((entry["step"], entry["eval_accuracy"]))

    if train_loss:
        steps, values = zip(*train_loss)
        plt.figure()
        plt.plot(steps, values, label="Train Loss", color="blue")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.legend()
        plt.savefig(plots_dir / "train_loss.png")
        plt.close()

    if eval_loss:
        steps, values = zip(*eval_loss)
        plt.figure()
        plt.plot(steps, values, label="Eval Loss", color="red")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Validation Loss")
        plt.legend()
        plt.savefig(plots_dir / "eval_loss.png")
        plt.close()

    if eval_acc:
        steps, values = zip(*eval_acc)
        plt.figure()
        plt.plot(steps, values, label="Accuracy", color="green")
        plt.xlabel("Step")
        plt.ylabel("Accuracy")
        plt.title("Validation Accuracy")
        plt.legend()
        plt.savefig(plots_dir / "eval_accuracy.png")
        plt.close()

    log.info(f"Графики сохранены в {plots_dir}")


def prepare_dataframe(dataset_path: str, random_state: int = 42) -> pd.DataFrame:
    base_path = Path(dataset_path)
    file_names: List[str] = []
    labels: List[str] = []

    patterns = ["*/*/*.jpeg", "*/*/*.jpg", "*/*/*.png"]
    found_files = []
    for pattern in patterns:
        found_files.extend(list(base_path.rglob(pattern)))

    if not found_files:
        raise FileNotFoundError(f"Изображения не найдены в папке {dataset_path}")

    log.info(f"Найдено {len(found_files)} файлов.")

    for file_path in found_files:
        label = file_path.parent.name
        labels.append(label)
        file_names.append(str(file_path))

    df = pd.DataFrame.from_dict({"image": file_names, "label": labels})

    y = df[["label"]]
    df_features = df.drop(["label"], axis=1)
    ros = RandomOverSampler(random_state=random_state)
    df_resampled, y_resampled = ros.fit_resample(df_features, y)
    df_resampled["label"] = y_resampled

    return df_resampled


def get_transforms(processor):
    image_mean, image_std = processor.image_mean, processor.image_std
    size = processor.size["height"]
    normalize = Normalize(mean=image_mean, std=image_std)

    _train_transforms = Compose(
        [
            Resize((size, size)),
            RandomRotation(20),
            RandomAdjustSharpness(2),
            ToTensor(),
            normalize,
        ]
    )

    _val_transforms = Compose(
        [
            Resize((size, size)),
            ToTensor(),
            normalize,
        ]
    )

    def train_transforms(examples):
        examples["pixel_values"] = [
            _train_transforms(image.convert("RGB")) for image in examples["image"]
        ]
        return examples

    def val_transforms(examples):
        examples["pixel_values"] = [
            _val_transforms(image.convert("RGB")) for image in examples["image"]
        ]
        return examples

    return train_transforms, val_transforms


def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    predictions = eval_pred.predictions
    label_ids = eval_pred.label_ids
    predicted_labels = predictions.argmax(axis=1)
    acc_score = accuracy.compute(predictions=predicted_labels, references=label_ids)[
        "accuracy"
    ]
    return {"accuracy": acc_score}


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def train_model(cfg: DictConfig):
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    git_commit_id = get_git_commit_id()

    with mlflow.start_run(run_name=f"train_{git_commit_id}") as run:
        log.info(f"MLflow Run ID: {run.info.run_id}")

        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))
        mlflow.set_tag("git_commit_id", git_commit_id)

        log.info(f"Загрузка данных из {cfg.data.dataset_path}")
        df = prepare_dataframe(cfg.data.dataset_path, cfg.random_seed)

        dataset = Dataset.from_pandas(df).cast_column("image", Image())
        labels_list = sorted(list(set(df["label"].unique())))
        label2id = {label: i for i, label in enumerate(labels_list)}
        id2label = {i: label for i, label in enumerate(labels_list)}

        ClassLabels = ClassLabel(num_classes=len(labels_list), names=labels_list)

        def map_label2id(example):
            example["label"] = ClassLabels.str2int(example["label"])
            return example

        dataset = dataset.map(map_label2id, batched=True)
        dataset = dataset.cast_column("label", ClassLabels)

        split_dataset = dataset.train_test_split(
            test_size=cfg.data.test_size, shuffle=True, stratify_by_column="label"
        )
        train_data = split_dataset["train"]
        test_data = split_dataset["test"]

        model_name = cfg.model.name
        processor = ViTImageProcessor.from_pretrained(model_name)
        model = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=len(labels_list),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )

        train_trans_func, val_trans_func = get_transforms(processor)
        train_data.set_transform(train_trans_func)
        test_data.set_transform(val_trans_func)

        training_args = TrainingArguments(
            output_dir=cfg.training.output_dir,
            logging_dir=f"{cfg.training.output_dir}/logs",
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=cfg.model.learning_rate,
            per_device_train_batch_size=cfg.training.batch_size,
            per_device_eval_batch_size=cfg.training.batch_size,
            num_train_epochs=cfg.training.num_train_epochs,
            weight_decay=cfg.model.weight_decay,
            warmup_steps=cfg.training.warmup_steps,
            remove_unused_columns=False,
            save_total_limit=cfg.training.save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            report_to="mlflow",
            run_name=f"train_{git_commit_id}",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=test_data,
            data_collator=collate_fn,
            compute_metrics=compute_metrics,
            tokenizer=processor,
        )

        log.info("Начало обучения...")
        trainer.train()

        log.info("Финальная оценка...")
        metrics = trainer.evaluate()
        log.info(f"Метрики: {metrics}")

        mlflow.log_metrics(metrics)

        save_path = f"{cfg.training.output_dir}/final_model"
        trainer.save_model(save_path)
        processor.save_pretrained(save_path)

        save_static_plots(trainer.state.log_history, "plots")
