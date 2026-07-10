from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix,
                             ConfusionMatrixDisplay, roc_curve, precision_recall_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 2074337326
# Amostra fixa usada apenas para acelerar o Credit Card Fraud, mantendo todos os casos de fraude.
NEGATIVE_SAMPLE_SIZE = 15000
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
FIGURES_DIR = BASE_DIR / "figures"
REPORTS_DIR = BASE_DIR / "reports"

DATASETS = {
    "Credit Card Fraud": {"file": DATA_DIR / "creditcard.csv.zip", "target": "Class", "positive": 1},
    "Bank Marketing": {"file": DATA_DIR / "bank.csv", "target": "deposit", "positive": "yes"},
}

MODELS = {
    "Logistic Regression": lambda: LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=40, max_depth=8, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE),
    "Decision Tree": lambda: DecisionTreeClassifier(max_depth=10, class_weight="balanced", random_state=RANDOM_STATE),
}


def target_label_names(dataset_name):
    if dataset_name == "Credit Card Fraud":
        return ["Transação normal", "Fraude"]
    if dataset_name == "Bank Marketing":
        return ["Não aderiu", "Aderiu"]
    return ["Caso negativo", "Caso positivo"]


def load_dataset(name, cfg):
    path = cfg["file"]
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            with z.open(z.namelist()[0]) as f:
                df = pd.read_csv(f)
    else:
        df = pd.read_csv(path, sep=None, engine="python")
    y = (df[cfg["target"]] == cfg["positive"]).astype(int)
    # Para manter a execução rápida e reprodutível no dataset Credit Card Fraud,
    # usa-se todos os registos positivos e uma amostra de negativos.
    if name == "Credit Card Fraud" and len(df) > 60000:
        pos = df[y == 1]
        neg = df[y == 0].sample(n=NEGATIVE_SAMPLE_SIZE, random_state=RANDOM_STATE)
        df = pd.concat([pos, neg]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
        y = (df[cfg["target"]] == cfg["positive"]).astype(int)
    X = df.drop(columns=[cfg["target"]])
    return X, y


def make_preprocessor(X):
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical))
    return ColumnTransformer(transformers)


def evaluate_dataset(dataset_name, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)
    pre = make_preprocessor(X)
    rows, probabilities, predictions, fitted = [], {}, {}, {}
    for model_name, factory in MODELS.items():
        pipe = Pipeline([("preprocess", pre), ("model", factory())])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        if hasattr(pipe[-1], "predict_proba"):
            prob = pipe.predict_proba(X_test)[:, 1]
        else:
            scores = pipe.decision_function(X_test)
            prob = (scores - scores.min()) / (scores.max() - scores.min())
        rows.append({
            "dataset": dataset_name,
            "model": model_name,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1_score": f1_score(y_test, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, prob),
            "avg_precision": average_precision_score(y_test, prob),
        })
        probabilities[model_name] = prob
        predictions[model_name] = pred
        fitted[model_name] = pipe
    return pd.DataFrame(rows), X_test, y_test, predictions, probabilities, fitted


def save_figures(all_results, per_dataset):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for p in FIGURES_DIR.glob("*.png"):
        p.unlink()

    pivot = all_results.pivot(index="model", columns="dataset", values="f1_score")
    ax = pivot.plot(kind="bar", figsize=(8, 4.5))
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1-score")
    ax.set_xlabel("Modelo")
    ax.set_title("Comparação dos 3 modelos por dataset")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_f1_comparison.png", dpi=200)
    plt.close()

    # Matriz de confusão do melhor modelo em cada dataset, com nomes claros.
    datasets_order = ["Bank Marketing", "Credit Card Fraud"]
    available = [d for d in datasets_order if d in per_dataset]
    if not available:
        available = list(per_dataset.keys())
    fig, axes = plt.subplots(1, len(available), figsize=(8 * len(available), 5))
    if len(available) == 1:
        axes = [axes]
    for ax, ds in zip(axes, available):
        best = all_results[all_results["dataset"] == ds].sort_values("f1_score", ascending=False).iloc[0]
        model = best["model"]
        X_test, y_test, preds, probs, _ = per_dataset[ds]
        labels = target_label_names(ds)
        disp = ConfusionMatrixDisplay(confusion_matrix(y_test, preds[model]), display_labels=labels)
        disp.plot(values_format="d", ax=ax, colorbar=False)
        ax.set_title(f"{ds}\n{model}")
        ax.set_xlabel("Previsão do modelo")
        ax.set_ylabel("Valor real")
        ax.tick_params(axis='x', labelrotation=15)
    fig.suptitle("Matrizes de confusão dos melhores modelos")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix_best_model.png", dpi=200)
    plt.close()

    # Curvas ROC e Precision-Recall para o dataset Credit Card Fraud, que é o caso de fraude direta.
    ds = "Credit Card Fraud" if "Credit Card Fraud" in per_dataset else list(per_dataset.keys())[0]
    X_test, y_test, preds, probs, _ = per_dataset[ds]

    plt.figure(figsize=(6, 4.5))
    for name, prob in probs.items():
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc = roc_auc_score(y_test, prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(f"Curva ROC - {ds}")
    plt.xlabel("Taxa de falsos positivos")
    plt.ylabel("Taxa de verdadeiros positivos")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_curve_models.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4.5))
    for name, prob in probs.items():
        precision, recall, _ = precision_recall_curve(y_test, prob)
        ap = average_precision_score(y_test, prob)
        plt.plot(recall, precision, label=f"{name} (AP={ap:.3f})")
    plt.title(f"Curva Precision-Recall - {ds}")
    plt.xlabel("Recall - positivos encontrados")
    plt.ylabel("Precision - positivos corretos")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "precision_recall_curve_models.png", dpi=200)
    plt.close()

    counts = []
    for name, cfg in DATASETS.items():
        _, y = load_dataset(name, cfg)
        vc = y.value_counts().sort_index()
        counts.append({"dataset": name, "Casos negativos": int(vc.get(0,0)), "Casos positivos": int(vc.get(1,0))})
    ax = pd.DataFrame(counts).set_index("dataset").plot(kind="bar", figsize=(7, 4))
    plt.title("Distribuição dos casos por dataset")
    plt.ylabel("Registos")
    plt.xlabel("Dataset")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_distribution.png", dpi=200)
    plt.close()


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    result_frames, per_dataset = [], {}
    summaries = []
    for name, cfg in DATASETS.items():
        X, y = load_dataset(name, cfg)
        summaries.append({"dataset": name, "rows": len(X), "features": X.shape[1], "positive_rate": float(y.mean())})
        res, X_test, y_test, preds, probs, fitted = evaluate_dataset(name, X, y)
        result_frames.append(res)
        per_dataset[name] = (X_test, y_test, preds, probs, fitted)
    results = pd.concat(result_frames, ignore_index=True).sort_values(["dataset", "f1_score"], ascending=[True, False])
    results.to_csv(REPORTS_DIR / "results_summary.csv", index=False)
    pd.DataFrame(summaries).to_csv(REPORTS_DIR / "datasets_summary.csv", index=False)
    save_figures(results, per_dataset)
    md = results.to_markdown(index=False, floatfmt=".4f")
    (REPORTS_DIR / "results_summary.md").write_text("# Resultados\n\n" + md + "\n", encoding="utf-8")
    print(results.to_string(index=False))

if __name__ == "__main__":
    main()
