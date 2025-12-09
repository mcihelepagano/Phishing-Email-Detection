import json
import os
import re
from collections import Counter
from typing import Callable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


class SequenceDataset(Dataset):
    """Holds pre-tokenized/padded sequences and labels."""

    def __init__(self, sequences: torch.Tensor, labels: list[int]):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class TextCNN(nn.Module):
    """Lightweight 1D CNN for text classification."""

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embedding_dim: int = 64,
        num_filters: int = 64,
        filter_sizes: tuple[int, ...] = (3, 4, 5),
        hidden_dim: int = 128,
        dropout: float = 0.3
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embedding_dim, num_filters, kernel_size=f) for f in filter_sizes]
        )
        conv_output_dim = num_filters * len(filter_sizes)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(conv_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        embedded = embedded.transpose(1, 2)  # (batch, embed_dim, seq_len)
        conv_results = [torch.relu(conv(embedded)) for conv in self.convs]
        pooled = [torch.max(c, dim=2).values for c in conv_results]
        concat = torch.cat(pooled, dim=1)
        return self.fc(concat)


class FocalLoss(nn.Module):
    """Focal loss to focus on harder examples (for imbalanced data)."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()


class DLTrainer:
    def __init__(
        self,
        datamanager,
        save_dir: str | None = None
    ):
        self.dm = datamanager
        self.save_dir = save_dir or os.path.join("models", "text_cnn_phishing")
        os.makedirs(self.save_dir, exist_ok=True)
        self.hf_save_dir = os.path.join("models", "distilbert_phishing")
        os.makedirs(self.hf_save_dir, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------
    # Data preparation helpers
    # -----------------------------
    def _ensure_text_column(self):
        if self.dm.df is None:
            raise ValueError("Please load the dataset first.")

        df = self.dm.df
        if "text_combined" not in df.columns:
            subject_series = df["Subject"].fillna("") if "Subject" in df.columns else pd.Series([""] * len(df))
            body_series = df["Body"].fillna("") if "Body" in df.columns else pd.Series([""] * len(df))
            df["text_combined"] = (subject_series.astype(str) + " " + body_series.astype(str)).str.strip()
        return df

    @staticmethod
    def _encode_labels(labels):
        unique_labels = sorted(set([str(label) for label in labels]))
        label2id = {label: idx for idx, label in enumerate(unique_labels)}
        encoded = [label2id[str(label)] for label in labels]
        id2label = {idx: label for label, idx in label2id.items()}
        return encoded, label2id, id2label

    @staticmethod
    def _tokenize(text: str):
        # Simple tokenization that avoids heavy dependencies
        return re.findall(r"\b\w+\b", str(text).lower())

    def _build_vocab(self, texts: list[str], max_vocab: int):
        counter = Counter()
        for txt in texts:
            counter.update(self._tokenize(txt))
        most_common = counter.most_common(max_vocab - 2)  # reserve PAD/UNK
        vocab = {"<PAD>": 0, "<UNK>": 1}
        vocab.update({word: idx + 2 for idx, (word, _) in enumerate(most_common)})
        return vocab

    def _texts_to_tensor(
        self,
        texts: list[str],
        vocab: dict,
        max_length: int
    ) -> torch.Tensor:
        unk_idx = vocab.get("<UNK>", 1)
        pad_idx = vocab.get("<PAD>", 0)
        sequences = []
        for txt in texts:
            tokens = self._tokenize(txt)
            ids = [vocab.get(tok, unk_idx) for tok in tokens[:max_length]]
            if len(ids) < max_length:
                ids += [pad_idx] * (max_length - len(ids))
            sequences.append(ids)
        return torch.tensor(sequences, dtype=torch.long)

    def _build_dataloaders(
        self,
        sequences: torch.Tensor,
        labels: list[int],
        batch_size: int
    ):
        splits = self.dm.ensure_split()
        train_idx = splits.get("train", np.array([], dtype=int))
        val_idx = splits.get("val", np.array([], dtype=int))

        # Fallback: if validation split is empty, reuse test as validation
        if val_idx.size == 0:
            val_idx = splits.get("test", np.array([], dtype=int))

        train_tensor_idx = torch.as_tensor(train_idx, dtype=torch.long)
        val_tensor_idx = torch.as_tensor(val_idx, dtype=torch.long)

        y_train = [labels[i] for i in train_idx]
        train_sequences = sequences[train_tensor_idx]
        val_sequences = sequences[val_tensor_idx]
        y_val = [labels[i] for i in val_idx]

        class_counts = np.bincount(y_train)
        class_weights = 1.0 / np.maximum(class_counts, 1)
        sample_weights = [class_weights[label] for label in y_train]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

        train_dataset = SequenceDataset(train_sequences, y_train)
        val_dataset = SequenceDataset(val_sequences, y_val)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        return train_loader, val_loader, class_weights, y_val

    def _build_hf_dataloaders(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer,
        batch_size: int,
        max_length: int,
        sample_limit: int | None = None
    ):
        splits = self.dm.ensure_split()
        train_idx = splits.get("train", np.array([], dtype=int))
        val_idx = splits.get("val", np.array([], dtype=int))
        if val_idx.size == 0:
            val_idx = splits.get("test", np.array([], dtype=int))

        if sample_limit and sample_limit > 0 and len(train_idx) > sample_limit:
            labels_train = [labels[i] for i in train_idx]
            stratify = labels_train if len(set(labels_train)) > 1 else None
            sampled_idx, _ = train_test_split(
                train_idx,
                train_size=sample_limit,
                random_state=42,
                stratify=stratify
            )
            train_idx = np.array(sampled_idx, dtype=int)

        train_texts = [texts[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        y_train = [labels[i] for i in train_idx]
        y_val = [labels[i] for i in val_idx]

        class_counts = np.bincount(y_train)
        class_weights = 1.0 / np.maximum(class_counts, 1)
        sample_weights = [class_weights[label] for label in y_train]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

        def collate_fn(batch):
            batch_texts, batch_labels = zip(*batch)
            enc = tokenizer(
                list(batch_texts),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            enc["labels"] = torch.tensor(batch_labels, dtype=torch.long)
            return enc

        class TextDataset(Dataset):
            def __init__(self, data_texts, data_labels):
                self.data_texts = data_texts
                self.data_labels = data_labels

            def __len__(self):
                return len(self.data_labels)

            def __getitem__(self, idx):
                return self.data_texts[idx], self.data_labels[idx]

        train_loader = DataLoader(
            TextDataset(train_texts, y_train),
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=0
        )
        val_loader = DataLoader(
            TextDataset(val_texts, y_val),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
        return train_loader, val_loader, class_weights, y_val

    # -----------------------------
    # Training and evaluation
    # -----------------------------
    def train_text_cnn(
        self,
        epochs: int = 4,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        max_length: int = 200,
        max_vocab_size: int = 20000,
        embedding_dim: int = 64,
        num_filters: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        focal_gamma: float = 2.0,
        train_sample_limit: int = 80000,
        scoring: str = "cross_entropy",
        use_genetic_threshold: bool = False
    ):
        """
        Train a lightweight text CNN instead of DistilBERT.
        scoring: 'cross_entropy' or 'focal' to choose the loss.
        use_genetic_threshold: when True (and binary), run a tiny GA to find the best decision threshold.
        """
        df = self._ensure_text_column()

        labels_raw = df["Label"]
        if labels_raw.isnull().any():
            df = df.dropna(subset=["Label"])
            labels_raw = df["Label"]

        texts = df["text_combined"].fillna("").astype(str).tolist()

        # Limit sample size to keep training feasible on modest hardware
        if train_sample_limit and train_sample_limit > 0 and len(texts) > train_sample_limit:
            stratify = labels_raw.tolist() if len(set(labels_raw)) > 1 else None
            sample_idx, _ = train_test_split(
                np.arange(len(texts)),
                train_size=train_sample_limit,
                random_state=42,
                stratify=stratify
            )
            texts = [texts[i] for i in sample_idx]
            labels_raw = labels_raw.iloc[sample_idx]
            print(f"Sampled {len(texts)} examples to keep training lightweight.")

        encoded_labels, label2id, id2label = self._encode_labels(labels_raw.tolist())

        vocab = self._build_vocab(texts, max_vocab_size)
        sequences = self._texts_to_tensor(texts, vocab, max_length)

        train_loader, val_loader, class_weights, val_labels = self._build_dataloaders(
            sequences, encoded_labels, batch_size
        )

        model = TextCNN(
            vocab_size=len(vocab),
            num_classes=len(label2id),
            embedding_dim=embedding_dim,
            num_filters=num_filters,
            hidden_dim=hidden_dim,
            dropout=dropout
        ).to(self.device)

        weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=self.device)
        if scoring == "focal":
            criterion: Callable = FocalLoss(gamma=focal_gamma, weight=weight_tensor)
        else:
            criterion = nn.CrossEntropyLoss(weight=weight_tensor)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
            for batch_sequences, batch_labels in progress:
                batch_sequences = batch_sequences.to(self.device)
                batch_labels = torch.tensor(batch_labels, dtype=torch.long, device=self.device)

                optimizer.zero_grad()
                logits = model(batch_sequences)
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                progress.set_postfix({"loss": f"{loss.item():.4f}"})
            print(f"Epoch {epoch + 1} - average training loss: {epoch_loss / max(len(train_loader), 1):.4f}")

        metrics, val_probs = self.evaluate(model, val_loader, id2label, return_probs=True)

        tuned_threshold = None
        if use_genetic_threshold and len(id2label) == 2 and val_probs is not None:
            tuned_threshold = self._genetic_search_threshold(val_probs, val_labels)
            metrics = self._apply_threshold(metrics, val_probs, val_labels, id2label, tuned_threshold)

        self._print_metrics(metrics)
        self._save_model(
            model,
            vocab,
            label2id,
            config={
                "max_length": max_length,
                "embedding_dim": embedding_dim,
                "num_filters": num_filters,
                "hidden_dim": hidden_dim,
                "dropout": dropout,
                "scoring": scoring,
                "focal_gamma": focal_gamma
            },
            threshold=tuned_threshold
        )
        return metrics

    def fine_tune_distilbert(
        self,
        epochs: int = 2,
        learning_rate: float = 5e-5,
        batch_size: int = 8,
        max_length: int = 256,
        model_checkpoint: str = "distilbert-base-uncased",
        train_sample_limit: int = 0
    ):
        df = self._ensure_text_column()

        labels_raw = df["Label"]
        if labels_raw.isnull().any():
            df = df.dropna(subset=["Label"])
            labels_raw = df["Label"]

        texts = df["text_combined"].fillna("").astype(str).tolist()
        encoded_labels, label2id, id2label = self._encode_labels(labels_raw.tolist())

        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        config = AutoConfig.from_pretrained(
            model_checkpoint,
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id
        )
        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, config=config)
        model.to(self.device)

        train_loader, val_loader, class_weights, val_labels = self._build_hf_dataloaders(
            texts, encoded_labels, tokenizer, batch_size, max_length, train_sample_limit
        )

        class_weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=self.device)
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=max(1, int(0.1 * total_steps)),
            num_training_steps=total_steps
        )
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weight_tensor)

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
            for batch in progress:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                optimizer.zero_grad()
                outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                logits = outputs.logits
                loss = loss_fn(logits, batch["labels"])
                loss.backward()
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item()
                progress.set_postfix({"loss": f"{loss.item():.4f}"})
            print(f"Epoch {epoch + 1} - average training loss: {epoch_loss / max(len(train_loader), 1):.4f}")

        metrics = self.evaluate_hf(model, val_loader, id2label, return_probs=False)
        self._print_metrics(metrics)
        self._save_hf_model(model, tokenizer, label2id, max_length)
        return metrics

    def evaluate_hf(self, model, dataloader, id2label, return_probs: bool = False):
        model.eval()
        all_labels, all_preds, prob_batches = [], [], []

        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)

                all_labels.extend(batch["labels"].cpu().tolist())
                all_preds.extend(preds.cpu().tolist())
                prob_batches.append(probs.cpu().numpy())

        prob_matrix = np.concatenate(prob_batches, axis=0) if prob_batches else np.empty((0, len(id2label)))
        metrics = self._compute_metrics(all_labels, all_preds, prob_matrix, id2label)
        if return_probs:
            return metrics, prob_matrix
        return metrics

    def evaluate(self, model, dataloader, id2label, return_probs: bool = False):
        model.eval()
        all_labels, all_preds, prob_batches = [], [], []

        with torch.no_grad():
            for batch_sequences, batch_labels in dataloader:
                batch_sequences = batch_sequences.to(self.device)
                batch_labels_tensor = torch.tensor(batch_labels, dtype=torch.long, device=self.device)
                logits = model(batch_sequences)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)

                all_labels.extend(batch_labels_tensor.cpu().tolist())
                all_preds.extend(preds.cpu().tolist())
                prob_batches.append(probs.cpu().numpy())

        prob_matrix = np.concatenate(prob_batches, axis=0) if prob_batches else np.empty((0, len(id2label)))
        metrics = self._compute_metrics(all_labels, all_preds, prob_matrix, id2label)
        if return_probs:
            return metrics, prob_matrix
        return metrics

    @staticmethod
    def _compute_metrics(labels, preds, prob_matrix, id2label):
        average = "binary" if len(id2label) == 2 else "weighted"
        accuracy = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average=average, zero_division=0
        )

        roc_auc = None
        pr_auc = None
        try:
            if len(id2label) == 2 and prob_matrix.shape[1] >= 2:
                roc_auc = roc_auc_score(labels, prob_matrix[:, 1])
                pr_auc = average_precision_score(labels, prob_matrix[:, 1])
            elif prob_matrix.size > 0:
                roc_auc = roc_auc_score(labels, prob_matrix, multi_class="ovr")
                pr_auc = average_precision_score(labels, prob_matrix, average="macro")
        except Exception:
            roc_auc = None
            pr_auc = None

        target_names = [id2label[idx] for idx in sorted(id2label)]
        report_text = classification_report(
            labels,
            preds,
            target_names=target_names,
            zero_division=0
        )

        support_per_class = {
            id2label[idx]: int(sum(1 for label in labels if label == idx))
            for idx in sorted(id2label)
        }

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "support": support_per_class,
            "report": report_text
        }

    @staticmethod
    def _print_metrics(metrics):
        print("--- Evaluation Metrics (DL) ---")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1']:.4f}")
        roc = metrics.get("roc_auc")
        if roc is not None:
            print(f"ROC-AUC:   {roc:.4f}")
        else:
            print("ROC-AUC:   not available for this run")
        pr = metrics.get("pr_auc")
        if pr is not None:
            print(f"PR-AUC:    {pr:.4f}")
        else:
            print("PR-AUC:    not available for this run")
        if "threshold" in metrics:
            print(f"Decision threshold (GA tuned): {metrics['threshold']:.3f}")
        print("Per-class support:")
        for label, count in metrics["support"].items():
            print(f"  {label}: {count}")
        print("Detailed classification report:")
        print(metrics["report"])

    @staticmethod
    def _genetic_search_threshold(prob_matrix, labels, population_size: int = 12, generations: int = 8):
        """Tiny GA to pick a decision threshold that maximizes F1 on validation."""
        population = np.random.uniform(0.25, 0.75, size=population_size)
        labels_array = np.array(labels)

        def fitness(thr):
            preds = (prob_matrix[:, 1] >= thr).astype(int)
            return f1_score(labels_array, preds, zero_division=0)

        for _ in range(generations):
            scores = np.array([fitness(thr) for thr in population])
            elite_idx = scores.argsort()[-4:]
            elite = population[elite_idx]
            children = []
            while len(children) + len(elite) < population_size:
                p1, p2 = np.random.choice(elite, 2)
                child = (p1 + p2) / 2 + np.random.normal(0, 0.02)
                child = float(np.clip(child, 0.05, 0.95))
                children.append(child)
            population = np.concatenate([elite, children])
        scores = np.array([fitness(thr) for thr in population])
        best_thr = float(population[np.argmax(scores)])
        return best_thr

    def _apply_threshold(self, base_metrics, prob_matrix, labels, id2label, threshold: float):
        preds = (prob_matrix[:, 1] >= threshold).astype(int)
        metrics = self._compute_metrics(labels, preds, prob_matrix, id2label)
        metrics["threshold"] = threshold
        return metrics

    # -----------------------------
    # Persistence
    # -----------------------------
    def _save_model(self, model, vocab, label2id, config: dict, threshold: float | None = None):
        torch.save(model.state_dict(), os.path.join(self.save_dir, "model.pt"))
        with open(os.path.join(self.save_dir, "vocab.json"), "w", encoding="utf-8") as fp:
            json.dump(vocab, fp, ensure_ascii=False)
        with open(os.path.join(self.save_dir, "label_mapping.json"), "w", encoding="utf-8") as fp:
            json.dump(label2id, fp, indent=2)
        with open(os.path.join(self.save_dir, "config.json"), "w", encoding="utf-8") as fp:
            json.dump(config, fp, indent=2)
        if threshold is not None:
            with open(os.path.join(self.save_dir, "threshold.json"), "w", encoding="utf-8") as fp:
                json.dump({"threshold": threshold}, fp, indent=2)
        print(f"Model artifacts saved to {self.save_dir}")

    def _save_hf_model(self, model, tokenizer, label2id, max_length: int):
        model.save_pretrained(self.hf_save_dir)
        tokenizer.save_pretrained(self.hf_save_dir)
        with open(os.path.join(self.hf_save_dir, "label_mapping.json"), "w", encoding="utf-8") as fp:
            json.dump(label2id, fp, indent=2)
        with open(os.path.join(self.hf_save_dir, "config.json"), "w", encoding="utf-8") as fp:
            json.dump({"max_length": max_length}, fp, indent=2)
        print(f"Saved DistilBERT model to {self.hf_save_dir}")

    def _load_vocab(self):
        vocab_path = os.path.join(self.save_dir, "vocab.json")
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"No vocab found at {vocab_path}")
        with open(vocab_path, "r", encoding="utf-8") as fp:
            return json.load(fp)

    def load_model(self):
        try:
            vocab = self._load_vocab()
            with open(os.path.join(self.save_dir, "label_mapping.json"), "r", encoding="utf-8") as fp:
                label2id = json.load(fp)
            with open(os.path.join(self.save_dir, "config.json"), "r", encoding="utf-8") as fp:
                config = json.load(fp)
            threshold_path = os.path.join(self.save_dir, "threshold.json")
            threshold = None
            if os.path.exists(threshold_path):
                with open(threshold_path, "r", encoding="utf-8") as fp:
                    threshold_data = json.load(fp)
                    threshold = threshold_data.get("threshold")
        except FileNotFoundError:
            print(f"No saved DL model found in {self.save_dir}")
            return None, None, None, None, None

        id2label = {int(v): k for k, v in label2id.items()}
        model = TextCNN(
            vocab_size=len(vocab),
            num_classes=len(label2id),
            embedding_dim=config.get("embedding_dim", 64),
            num_filters=config.get("num_filters", 64),
            hidden_dim=config.get("hidden_dim", 128),
            dropout=config.get("dropout", 0.3)
        )
        model.load_state_dict(torch.load(os.path.join(self.save_dir, "model.pt"), map_location=self.device))
        model.to(self.device)

        print(f"Loaded DL model from {self.save_dir} (device: {self.device})")
        return model, vocab, id2label, config, threshold

    def _load_hf_model(self):
        if not os.path.exists(self.hf_save_dir):
            print(f"No saved DistilBERT model found in {self.hf_save_dir}")
            return None, None, None, None
        try:
            with open(os.path.join(self.hf_save_dir, "label_mapping.json"), "r", encoding="utf-8") as fp:
                label2id = json.load(fp)
        except FileNotFoundError:
            label2id = {}
        try:
            with open(os.path.join(self.hf_save_dir, "config.json"), "r", encoding="utf-8") as fp:
                extra_config = json.load(fp)
        except FileNotFoundError:
            extra_config = {}

        tokenizer = AutoTokenizer.from_pretrained(self.hf_save_dir)
        config = AutoConfig.from_pretrained(self.hf_save_dir)
        model = AutoModelForSequenceClassification.from_pretrained(self.hf_save_dir, config=config)
        model.to(self.device)
        id2label = {int(v): k for k, v in label2id.items()} if label2id else config.id2label
        max_length = extra_config.get("max_length", 256)
        return model, tokenizer, id2label, max_length

    def evaluate_saved_model(self, batch_size: int = 32):
        model, vocab, id2label, config, threshold = self.load_model()
        if model is None:
            return

        df = self._ensure_text_column()
        labels_raw = df["Label"]
        if labels_raw.isnull().any():
            df = df.dropna(subset=["Label"])
            labels_raw = df["Label"]

        texts = df["text_combined"].fillna("").astype(str).tolist()
        label2id = {label: idx for idx, label in id2label.items()}
        encoded_labels = [label2id.get(str(label), 0) for label in labels_raw.tolist()]

        sequences = self._texts_to_tensor(texts, vocab, config.get("max_length", 200))

        # Reuse dataloaders for evaluation only (we only keep the validation loader/labels)
        _, val_loader, _, val_labels = self._build_dataloaders(sequences, encoded_labels, batch_size)

        metrics, prob_matrix = self.evaluate(model, val_loader, id2label, return_probs=True)
        if threshold is not None and prob_matrix is not None and len(id2label) == 2:
            metrics = self._apply_threshold(metrics, prob_matrix, val_labels, id2label, threshold)
        self._print_metrics(metrics)
        return metrics

    def evaluate_saved_distilbert(self, batch_size: int = 8):
        model, tokenizer, id2label, max_length = self._load_hf_model()
        if model is None or tokenizer is None or id2label is None:
            return

        df = self._ensure_text_column()
        labels_raw = df["Label"]
        if labels_raw.isnull().any():
            df = df.dropna(subset=["Label"])
            labels_raw = df["Label"]

        texts = df["text_combined"].fillna("").astype(str).tolist()
        label2id = {label: idx for idx, label in id2label.items()} if isinstance(id2label, dict) else {}
        encoded_labels = [label2id.get(str(label), 0) for label in labels_raw.tolist()]

        _, val_loader, _, _ = self._build_hf_dataloaders(
            texts, encoded_labels, tokenizer, batch_size, max_length
        )
        metrics = self.evaluate_hf(model, val_loader, id2label, return_probs=False)
        self._print_metrics(metrics)
        return metrics
