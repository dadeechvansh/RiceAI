"""
============================================================
Rice Variety Classification - Training Pipeline (v2.0)
============================================================

Author  : Vansh Dadeech
Version : 2.0

Trains, tunes, evaluates, and compares multiple classifiers on the
Rice Variety dataset (Cammeo vs Osmancik), selects the best performing
model automatically, and persists all artifacts required by the rest
of the project (predict.py, app.py).

Outputs
-------
models/best_model.pkl
models/scaler.pkl
results/results.json
images/confusion_matrix.png
images/accuracy_comparison.png
images/correlation_heatmap.png
images/class_distribution.png
images/feature_distributions.png
images/feature_importance.png   (only if best model supports it)

Run
---
python train.py
"""

from __future__ import annotations

import json
import logging
import joblib
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# ============================================================
# Configuration
# ============================================================


@dataclass(frozen=True)
class Config:
    """Central configuration for the training pipeline."""

    data_path: Path = Path("data/riceClassification.csv")
    models_dir: Path = Path("models")
    results_dir: Path = Path("results")
    images_dir: Path = Path("images")

    target_column: str = "Class"
    test_size: float = 0.20
    random_state: int = 42
    cv_folds: int = 5

    # Columns that are pure identifiers / leakage risks and should be
    # dropped if present (defensive - dataset may or may not include one).
    columns_to_drop: list[str] = field(default_factory=lambda: ["id"])


CONFIG = Config()


# ============================================================
# Logging
# ============================================================


def setup_logging() -> logging.Logger:
    """Configure and return the module-level logger.

    Returns:
        A configured logger that writes to stdout with timestamps.
    """
    logger = logging.getLogger("rice_classifier")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


logger = setup_logging()


# ============================================================
# Utility: folder setup
# ============================================================


def create_project_directories(config: Config) -> None:
    """Create the models/results/images directories if missing.

    Args:
        config: Pipeline configuration holding target directory paths.
    """
    for directory in (config.models_dir, config.results_dir, config.images_dir):
        directory.mkdir(parents=True, exist_ok=True)
    logger.info("Project directories verified: models/, results/, images/")


# ============================================================
# Data loading & validation
# ============================================================


def load_dataset(config: Config) -> pd.DataFrame:
    """Load the rice dataset from disk with defensive error handling.

    Args:
        config: Pipeline configuration holding the dataset path.

    Returns:
        The loaded dataset as a DataFrame.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If the dataset is empty or missing the target column.
    """
    if not config.data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{config.data_path}'. "
            "Please verify the file path before running training."
        )

    df = pd.read_csv(config.data_path)

    if df.empty:
        raise ValueError("Loaded dataset is empty. Cannot proceed with training.")

    if config.target_column not in df.columns:
        raise ValueError(
            f"Target column '{config.target_column}' not found in dataset. "
            f"Available columns: {list(df.columns)}"
        )

    logger.info("Dataset loaded successfully | shape=%s", df.shape)
    return df


def validate_and_clean_dataset(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Run data-quality checks and clean the dataset.

    Checks for missing values, duplicate rows, and drops known
    non-feature identifier columns if present. Logs a full statistical
    summary and the target class distribution.

    Args:
        df: Raw input DataFrame.
        config: Pipeline configuration.

    Returns:
        A cleaned copy of the DataFrame.
    """
    df = df.copy()

    # Drop identifier-like columns if they exist (defensive, not assumed).
    existing_drop_cols = [c for c in config.columns_to_drop if c in df.columns]
    if existing_drop_cols:
        df = df.drop(columns=existing_drop_cols)
        logger.info("Dropped non-feature columns: %s", existing_drop_cols)

    # Missing values.
    missing_counts = df.isnull().sum()
    total_missing = int(missing_counts.sum())
    if total_missing > 0:
        logger.warning("Found %d missing values:\n%s", total_missing, missing_counts[missing_counts > 0])
        df = df.dropna()
        logger.info("Rows after dropping missing values: %d", len(df))
    else:
        logger.info("No missing values detected.")

    # Duplicates.
    duplicate_count = int(df.duplicated().sum())
    if duplicate_count > 0:
        df = df.drop_duplicates()
        logger.warning("Removed %d duplicate rows. New shape=%s", duplicate_count, df.shape)
    else:
        logger.info("No duplicate rows detected.")

    # Descriptive statistics.
    logger.info("Dataset statistics:\n%s", df.describe().to_string())

    # Target distribution.
    class_counts = df[config.target_column].value_counts()
    class_pct = (class_counts / len(df) * 100).round(2)
    logger.info(
        "Target class distribution:\n%s",
        pd.concat([class_counts, class_pct], axis=1, keys=["count", "percent"]).to_string(),
    )

    return df.reset_index(drop=True)


# ============================================================
# Exploratory Data Analysis
# ============================================================


def generate_eda_visuals(df: pd.DataFrame, config: Config) -> None:
    """Generate and save EDA visualizations: correlation heatmap,
    class distribution, and feature histograms.

    Args:
        df: Cleaned DataFrame (target column still as original labels
            or encoded integers - both work for these plots).
        config: Pipeline configuration.
    """
    numeric_df = df.select_dtypes(include=[np.number])

    # --- Correlation heatmap ---
    plt.figure(figsize=(9, 7))
    correlation = numeric_df.corr()
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(config.images_dir / "correlation_heatmap.png", dpi=300)
    plt.close()
    logger.info("Saved images/correlation_heatmap.png")

    # --- Class distribution ---
    plt.figure(figsize=(6, 5))
    sns.countplot(x=df[config.target_column].astype(str))
    plt.title("Target Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(config.images_dir / "class_distribution.png", dpi=300)
    plt.close()
    logger.info("Saved images/class_distribution.png")

    # --- Feature distribution histograms ---
    feature_cols = [c for c in numeric_df.columns if c != config.target_column]
    n_cols = 3
    n_rows = int(np.ceil(len(feature_cols) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.array(axes).reshape(-1)

    for idx, col in enumerate(feature_cols):
        sns.histplot(df[col], kde=True, ax=axes[idx], color="teal")
        axes[idx].set_title(f"Distribution: {col}")

    for idx in range(len(feature_cols), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.savefig(config.images_dir / "feature_distributions.png", dpi=300)
    plt.close()
    logger.info("Saved images/feature_distributions.png")


# ============================================================
# Preprocessing
# ============================================================


def preprocess_data(
    df: pd.DataFrame, config: Config
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, LabelEncoder, list[str]]:
    """Encode the target, split into train/test, and scale features.

    Args:
        df: Cleaned DataFrame.
        config: Pipeline configuration.

    Returns:
        A tuple of (X_train, X_test, y_train, y_test, scaler, encoder, feature_names).
    """
    df = df.copy()

    encoder = LabelEncoder()
    df[config.target_column] = encoder.fit_transform(df[config.target_column])
    logger.info("Label encoding map: %s", dict(zip(encoder.classes_, encoder.transform(encoder.classes_))))

    X = df.drop(columns=[config.target_column])
    y = df[config.target_column]
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )
    logger.info("Train/test split -> train=%d, test=%d", len(X_train), len(X_test))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    with open(config.models_dir / "scaler.pkl", "wb") as f:
        joblib.dump(scaler, f)
    logger.info("Saved models/scaler.pkl")

    return X_train_scaled, X_test_scaled, y_train.to_numpy(), y_test.to_numpy(), scaler, encoder, feature_names


# ============================================================
# Model definitions & hyperparameter search spaces
# ============================================================


def build_model_search_specs(config: Config) -> dict[str, dict[str, Any]]:
    """Define each model along with its tuning strategy and search space.

    Tuning strategy is chosen deliberately per model rather than
    blindly grid-searching everything:
      - Logistic Regression: fast, near-convex loss -> light grid search
        on regularization strength is enough; no need for randomized search.
      - Decision Tree: small, discrete search space -> GridSearchCV is cheap
        and exhaustive here.
      - KNN: small discrete search space -> GridSearchCV.
      - SVM (RBF): continuous C/gamma space, more expensive to fit ->
        RandomizedSearchCV to control compute cost.
      - Random Forest: large combinatorial space, expensive to fit ->
        RandomizedSearchCV to control compute cost.

    Args:
        config: Pipeline configuration (used for random_state).

    Returns:
        Mapping of model name -> {"estimator", "search_type", "param_grid", "n_iter"}.
    """
    rs = config.random_state

    return {
        "Logistic Regression": {
            "estimator": LogisticRegression(max_iter=2000, random_state=rs),
            "search_type": "grid",
            "param_grid": {
                "C": [0.01, 0.1, 1, 10, 100],
                "solver": ["lbfgs", "liblinear"],
            },
        },
        "Decision Tree": {
            "estimator": DecisionTreeClassifier(random_state=rs),
            "search_type": "grid",
            "param_grid": {
                "max_depth": [4, 6, 8, 10, 15, None],
                "min_samples_split": [2, 5, 10],
                "criterion": ["gini", "entropy"],
            },
        },
        "KNN": {
            "estimator": KNeighborsClassifier(),
            "search_type": "grid",
            "param_grid": {
                "n_neighbors": [3, 5, 7, 9, 11, 15],
                "weights": ["uniform", "distance"],
                "p": [1, 2],
            },
        },
        "SVM": {
            "estimator": SVC(probability=True, random_state=rs),
            "search_type": "random",
            "param_grid": {
                "C": [0.1, 1, 10, 50, 100],
                "gamma": ["scale", "auto", 0.01, 0.1, 1],
                "kernel": ["rbf", "linear"],
            },
            "n_iter": 12,
        },
        "Random Forest": {
            "estimator": RandomForestClassifier(random_state=rs),
            "search_type": "random",
            "param_grid": {
                "n_estimators": [100, 200, 300, 500],
                "max_depth": [None, 8, 12, 20],
                "min_samples_split": [2, 5, 10],
                "max_features": ["sqrt", "log2"],
            },
            "n_iter": 15,
        },
    }


# ============================================================
# Training & tuning
# ============================================================


def tune_model(
    name: str,
    spec: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: Config,
) -> Any:
    """Run the appropriate hyperparameter search for a single model.

    Args:
        name: Model display name (for logging).
        spec: Model search specification (estimator, search type, grid).
        X_train: Scaled training features.
        y_train: Training labels.
        config: Pipeline configuration.

    Returns:
        The best fitted estimator found by the search.
    """
    cv = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)

    if spec["search_type"] == "grid":
        search = GridSearchCV(
            estimator=spec["estimator"],
            param_grid=spec["param_grid"],
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
        )
    else:
        search = RandomizedSearchCV(
            estimator=spec["estimator"],
            param_distributions=spec["param_grid"],
            n_iter=spec.get("n_iter", 10),
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
            random_state=config.random_state,
        )

    start = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - start

    logger.info(
        "%-20s | best_cv_accuracy=%.4f | best_params=%s | time=%.1fs",
        name,
        search.best_score_,
        search.best_params_,
        elapsed,
    )

    return search.best_estimator_


def evaluate_model(
    name: str,
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: Config,
) -> dict[str, Any]:
    """Compute the full evaluation metric suite for a trained model.

    Args:
        name: Model display name.
        model: Fitted estimator.
        X_train: Scaled training features (for cross-validation scoring).
        y_train: Training labels.
        X_test: Scaled test features.
        y_test: True test labels.
        config: Pipeline configuration.

    Returns:
        Dictionary of metrics: accuracy, precision, recall, f1,
        cv_mean, cv_std, and the raw predictions.
    """
    predictions = model.predict(X_test)

    cv = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, average="weighted")), 4),
        "recall": round(float(recall_score(y_test, predictions, average="weighted")), 4),
        "f1_score": round(float(f1_score(y_test, predictions, average="weighted")), 4),
        "cv_mean_accuracy": round(float(cv_scores.mean()), 4),
        "cv_std_accuracy": round(float(cv_scores.std()), 4),
    }

    logger.info(
        "%-20s | acc=%.4f | precision=%.4f | recall=%.4f | f1=%.4f | cv=%.4f (+/-%.4f)",
        name,
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1_score"],
        metrics["cv_mean_accuracy"],
        metrics["cv_std_accuracy"],
    )

    return {"metrics": metrics, "predictions": predictions, "model": model}


# ============================================================
# Model selection
# ============================================================


def select_best_model(
    trained_models: dict[str, dict[str, Any]]
) -> tuple[str, dict[str, Any]]:
    """Automatically select the best model by test-set accuracy.

    Ties are broken by cross-validation mean accuracy to prefer
    the model that generalizes more consistently.

    Args:
        trained_models: Mapping of model name -> evaluation result dict.

    Returns:
        Tuple of (best_model_name, its evaluation result dict).
    """
    best_name = max(
        trained_models,
        key=lambda n: (
            trained_models[n]["metrics"]["accuracy"],
            trained_models[n]["metrics"]["cv_mean_accuracy"],
        ),
    )
    return best_name, trained_models[best_name]


# ============================================================
# Persistence
# ============================================================


def save_best_model(model: Any, config: Config) -> None:
    """Persist the best model to disk.

    Args:
        model: Fitted best estimator.
        config: Pipeline configuration.
    """
    with open(config.models_dir / "best_model.pkl", "wb") as f:
        joblib.dump(model, f)
    logger.info("Saved models/best_model.pkl")


def save_results_json(
    df_shape: tuple[int, int],
    feature_names: list[str],
    best_model_name: str,
    trained_models: dict[str, dict[str, Any]],
    config: Config,
) -> None:
    """Persist a structured summary of the training run to results.json.

    Args:
        df_shape: Shape of the cleaned dataset (rows, columns).
        feature_names: List of feature column names used for training.
        best_model_name: Name of the automatically selected best model.
        trained_models: Mapping of model name -> evaluation result dict.
        config: Pipeline configuration.
    """
    best_metrics = trained_models[best_model_name]["metrics"]

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_shape": {"rows": df_shape[0], "columns": df_shape[1]},
        "features": feature_names,
        "best_model": best_model_name,
        "best_model_accuracy": best_metrics["accuracy"],
        "best_model_cv_score": best_metrics["cv_mean_accuracy"],
        "models": {
            name: result["metrics"] for name, result in trained_models.items()
        },
    }

    with open(config.results_dir / "results.json", "w") as f:
        json.dump(results, f, indent=4)

    logger.info("Saved results/results.json")


# ============================================================
# Visualizations
# ============================================================


def plot_confusion_matrix(
    y_test: np.ndarray,
    predictions: np.ndarray,
    class_names: np.ndarray,
    best_model_name: str,
    config: Config,
) -> None:
    """Plot and save the confusion matrix for the best model.

    Args:
        y_test: True test labels.
        predictions: Predicted labels from the best model.
        class_names: Original class label names (from the LabelEncoder).
        best_model_name: Name of the best model (used in the title).
        config: Pipeline configuration.
    """
    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(f"Confusion Matrix ({best_model_name})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(config.images_dir / "confusion_matrix.png", dpi=300)
    plt.close()
    logger.info("Saved images/confusion_matrix.png")


def plot_accuracy_comparison(trained_models: dict[str, dict[str, Any]], config: Config) -> None:
    """Plot and save a bar chart comparing test accuracy across all models.

    Args:
        trained_models: Mapping of model name -> evaluation result dict.
        config: Pipeline configuration.
    """
    model_names = list(trained_models.keys())
    accuracies = [trained_models[name]["metrics"]["accuracy"] for name in model_names]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(model_names, accuracies, color=sns.color_palette("viridis", len(model_names)))
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.ylim(max(0.0, min(accuracies) - 0.05), 1.00)

    for bar, value in zip(bars, accuracies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.003,
            f"{value:.3f}",
            ha="center",
            fontsize=10,
        )

    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(config.images_dir / "accuracy_comparison.png", dpi=300)
    plt.close()
    logger.info("Saved images/accuracy_comparison.png")


def plot_feature_importance(
    model: Any, feature_names: list[str], best_model_name: str, config: Config
) -> None:
    """Plot and save feature importance if the model supports it.

    Args:
        model: Fitted best estimator.
        feature_names: List of feature column names.
        best_model_name: Name of the best model (used in the title).
        config: Pipeline configuration.
    """
    if not hasattr(model, "feature_importances_"):
        logger.info("%s does not expose feature_importances_; skipping plot.", best_model_name)
        return

    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]

    plt.figure(figsize=(8, 5))
    plt.bar(
        [feature_names[i] for i in order],
        importances[order],
        color="steelblue",
    )
    plt.title(f"Feature Importance ({best_model_name})")
    plt.ylabel("Importance")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(config.images_dir / "feature_importance.png", dpi=300)
    plt.close()
    logger.info("Saved images/feature_importance.png")


# ============================================================
# Main pipeline
# ============================================================


def main() -> None:
    """Run the full end-to-end training pipeline."""
    logger.info("=" * 60)
    logger.info("Rice Variety Classification - Training Pipeline v2.0")
    logger.info("=" * 60)

    config = CONFIG
    create_project_directories(config)

    # 1. Load & validate data.
    try:
        raw_df = load_dataset(config)
    except (FileNotFoundError, ValueError) as e:
        logger.error("Failed to load dataset: %s", e)
        sys.exit(1)

    clean_df = validate_and_clean_dataset(raw_df, config)

    # 2. EDA.
    logger.info("-" * 60)
    logger.info("Generating EDA visualizations...")
    generate_eda_visuals(clean_df, config)

    # 3. Preprocessing.
    logger.info("-" * 60)
    logger.info("Preprocessing data...")
    X_train, X_test, y_train, y_test, scaler, encoder, feature_names = preprocess_data(clean_df, config)

    # 4. Train & tune every model.
    logger.info("-" * 60)
    logger.info("Training and tuning models...")
    search_specs = build_model_search_specs(config)

    trained_models: dict[str, dict[str, Any]] = {}
    for name, spec in search_specs.items():
        try:
            best_estimator = tune_model(name, spec, X_train, y_train, config)
            result = evaluate_model(name, best_estimator, X_train, y_train, X_test, y_test, config)
            trained_models[name] = result
        except Exception as e:  # noqa: BLE001 - log and continue with remaining models
            logger.error("Training failed for %s: %s", name, e)

    if not trained_models:
        logger.error("No models were successfully trained. Aborting.")
        sys.exit(1)

    # 5. Select best model.
    logger.info("-" * 60)
    best_model_name, best_result = select_best_model(trained_models)
    logger.info(
        "Best model selected: %s (accuracy=%.4f, cv=%.4f)",
        best_model_name,
        best_result["metrics"]["accuracy"],
        best_result["metrics"]["cv_mean_accuracy"],
    )

    # 6. Classification report for best model.
    logger.info("-" * 60)
    logger.info(
        "Classification Report (%s):\n%s",
        best_model_name,
        classification_report(y_test, best_result["predictions"], target_names=encoder.classes_),
    )

    # 7. Persist artifacts.
    logger.info("-" * 60)
    save_best_model(best_result["model"], config)
    save_results_json(clean_df.shape, feature_names, best_model_name, trained_models, config)

    # 8. Visualizations.
    plot_confusion_matrix(y_test, best_result["predictions"], encoder.classes_, best_model_name, config)
    plot_accuracy_comparison(trained_models, config)
    plot_feature_importance(best_result["model"], feature_names, best_model_name, config)

    logger.info("=" * 60)
    logger.info("Training Completed Successfully")
    logger.info("Best Model : %s", best_model_name)
    logger.info("Accuracy   : %.4f", best_result["metrics"]["accuracy"])
    logger.info("=" * 60)
    logger.info("Saved files:")
    logger.info("  models/best_model.pkl")
    logger.info("  models/scaler.pkl")
    logger.info("  results/results.json")
    logger.info("  images/confusion_matrix.png")
    logger.info("  images/accuracy_comparison.png")
    logger.info("  images/correlation_heatmap.png")
    logger.info("  images/class_distribution.png")
    logger.info("  images/feature_distributions.png")
    logger.info("Project ready for deployment.")


if __name__ == "__main__":
    main()