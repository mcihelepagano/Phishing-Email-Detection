import os
from typing import Optional

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, average_precision_score, classification_report
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


class ModelManager:

    def __init__(self, datamanager):
        self.dm = datamanager
        self.model = None
        self.model_name = None
        self.models_dir = os.path.join("models")
        os.makedirs(self.models_dir, exist_ok=True)

    # -----------------------------
    # Data preparation
    # -----------------------------
    def get_features_and_labels(self):
        """Return feature matrix and labels, combining text vectors with engineered features."""
        if getattr(self.dm, "X_processed", None) is None:
            raise ValueError("Data not vectorized yet. Please preprocess first.")
        if "Label" not in self.dm.df.columns:
            raise ValueError("No 'Label' column found in dataset.")

        X_text = self.dm.X_processed
        df = self.dm.df

        # Combine text vectors with engineered numeric/categorical features if available
        extra_cols = getattr(self.dm, "encoded_feature_columns", []) or []
        valid_extra_cols = [col for col in extra_cols if col in df.columns]

        if valid_extra_cols:
            from scipy.sparse import hstack, csr_matrix
            import numpy as np

            extras = csr_matrix(df[valid_extra_cols].fillna(0).astype(np.float32).values)
            X = hstack([X_text, extras])
        else:
            X = X_text

        y = df["Label"]
        # Cache for reuse (e.g., cross-validation)
        self.X = X
        self.y = y
        return X, y

    # -----------------------------
    # Training
    # -----------------------------
    def train_model(self, model_type="RandomForest"):
        """Train a classical ML model (RandomForest, SGDClassifier, LinearSVM, or DecisionTree)."""
        X, y = self.get_features_and_labels()
        splits = self.dm.ensure_split()
        train_idx, test_idx = splits["train"], splits["test"]
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model_type = model_type.lower().replace(" ", "_")

        if model_type == "random_forest":
            self.model_name = "RandomForest"
            self.model = RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1
            )
        elif model_type == "sgd":
            self.model_name = "SGDClassifier"
            self.model = SGDClassifier(loss="log_loss", max_iter=1000, random_state=42)
        elif model_type == "linear_svm":
            self.model_name = "LinearSVM"
            self.model = LinearSVC(
                class_weight="balanced",
                max_iter=2000
            )
        elif model_type == "decision_tree":
            self.model_name = "DecisionTree"
            self.model = DecisionTreeClassifier(
                random_state=42,
                max_depth=20,
                min_samples_split=50,
                min_samples_leaf=20,
                class_weight="balanced"
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        print(f"\nTraining {self.model_name}...")
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        # Compute PR-AUC if the estimator can score probabilities or decision function
        pr_auc = None
        try:
            if hasattr(self.model, "predict_proba"):
                scores = self.model.predict_proba(X_test)[:, 1]
                pr_auc = average_precision_score(y_test, scores)
            elif hasattr(self.model, "decision_function"):
                scores = self.model.decision_function(X_test)
                pr_auc = average_precision_score(y_test, scores)
        except Exception:
            pr_auc = None

        print(f"\nAccuracy: {acc:.4f}\n")
        if pr_auc is not None:
            print(f"PR-AUC (Average Precision): {pr_auc:.4f}\n")
        print(classification_report(y_test, y_pred))

    # -----------------------------
    # Cross Validation
    # -----------------------------
    def cross_validate_model(self, model_type: Optional[str] = None, n_folds=5, use_multiple_metrics=False):
        """Perform k-fold cross-validation on the specified model type."""
        X, y = self.get_features_and_labels()

        # Derive model type: caller override > existing model_name > default RF
        if model_type is None:
            model_type = self.model_name or "random_forest"
        model_type = model_type.lower().replace(" ", "_")

        if model_type == "random_forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        elif model_type == "sgd":
            model = SGDClassifier(loss="log_loss", max_iter=1000, random_state=42)
        elif model_type == "linear_svm":
            model = LinearSVC(
                class_weight="balanced",
                max_iter=2000
            )
        elif model_type == "decision_tree":
            model = DecisionTreeClassifier(
                random_state=42,
                max_depth=20,
                min_samples_split=50,
                min_samples_leaf=20,
                class_weight="balanced"
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        if use_multiple_metrics:
            scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'average_precision']
            results = cross_validate(model, X, y, cv=kf, scoring=scoring, n_jobs=-1)

            print(f"\nCross-validation results for {model_type}:")
            print(f"Accuracy:  {results['test_accuracy'].mean():.4f} (+/- {results['test_accuracy'].std():.4f})")
            print(f"Precision: {results['test_precision'].mean():.4f} (+/- {results['test_precision'].std():.4f})")
            print(f"Recall:    {results['test_recall'].mean():.4f} (+/- {results['test_recall'].std():.4f})")
            print(f"F1 Score:  {results['test_f1'].mean():.4f} (+/- {results['test_f1'].std():.4f})")
            print(f"ROC-AUC:   {results['test_roc_auc'].mean():.4f} (+/- {results['test_roc_auc'].std():.4f})")
            print(f"PR-AUC:    {results['test_average_precision'].mean():.4f} (+/- {results['test_average_precision'].std():.4f})")

            return results
        else:
            results = cross_validate(model, X, y, cv=kf, scoring='accuracy', n_jobs=-1)
            scores = results['test_accuracy']

            print(f"\nCross-validation results for {model_type}:")
            print(f"Mean accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
            print(f"Individual fold scores: {scores}")

            return scores

    # -----------------------------
    # Saving and loading
    # -----------------------------
    def save_model(self):
        if self.model is None:
            print("No model trained yet.")
            return

        path = os.path.join(self.models_dir, f"{self.model_name}.joblib")
        joblib.dump(self.model, path)
        print(f"\nModel saved to {path}")

    def load_model(self, model_name):
        path = os.path.join(self.models_dir, f"{model_name}.joblib")

        if not os.path.exists(path):
            print(f"No saved model found at {path}")
            return

        self.model = joblib.load(path)
        self.model_name = model_name
        print(f"\nLoaded model: {model_name}")

    # -----------------------------
    # Continue training
    # -----------------------------
    def continue_training(self, evaluate: bool = True, test_size: float = 0.2):
        """Optionally evaluate on a holdout split, then refit the model on all data."""
        if self.model is None:
            print("No model loaded or trained.")
            return

        X, y = self.get_features_and_labels()

        splits = self.dm.ensure_split(test_size=test_size)
        train_idx, test_idx = splits["train"], splits["test"]

        if evaluate:
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            print(f"\nContinuing training of {self.model_name} with holdout evaluation ({test_size:.0%})...")
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            print(f"\nHoldout accuracy: {acc:.4f}\n")
            print(classification_report(y_test, y_pred))
        else:
            print(f"\nContinuing training of {self.model_name} on full data...")

        # Final fit on all data
        self.model.fit(X, y)
        print("Model updated.")
