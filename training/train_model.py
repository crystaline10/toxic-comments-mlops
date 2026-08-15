import os
import joblib
import pandas as pd
import wandb
import subprocess
import hashlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline


DATA_PATH = "data/train.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "toxic_comment_model.pkl")

WANDB_PROJECT = "toxic-comments"

def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        return "unknown"




CONFIG = {
    "max_features": 50000,
    "ngram_range": (1, 2),
    "C": 1.0,
    "class_weight": "balanced",
    "test_size": 0.2,
    "random_state": 42,
    "git_commit": get_git_commit(),
}

LABEL_COLUMNS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]


def load_data():
    df = pd.read_csv(DATA_PATH)

    X = df["comment_text"]
    y = df[LABEL_COLUMNS]

    return X, y


def build_model():
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=CONFIG["max_features"],
                    stop_words="english",
                    ngram_range=CONFIG["ngram_range"],
                ),
            ),
            (
                "classifier",
                MultiOutputClassifier(
                    LogisticRegression(
                        max_iter=1000,
                        C=CONFIG["C"],
                        class_weight=CONFIG["class_weight"],
                    )
                ),
            ),
        ]
    )

    return pipeline


def main():
    run = wandb.init(
        project=WANDB_PROJECT,
        config=CONFIG,
        name="baseline-logistic-regression",
    )

    print("Loading dataset...")
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=CONFIG["test_size"],
        random_state=CONFIG["random_state"],
    )

    print("Training model...")
    model = build_model()
    model.fit(X_train, y_train)

    print("Evaluating model...")
    predictions = model.predict(X_test)

    exact_match_accuracy = accuracy_score(y_test, predictions)
    micro_f1 = f1_score(y_test, predictions, average="micro")
    macro_f1 = f1_score(y_test, predictions, average="macro")

    wandb.log(
        {
            "exact_match_accuracy": exact_match_accuracy,
            "micro_f1": micro_f1,
            "macro_f1": macro_f1
        }
    )

    print(f"Exact Match Accuracy: {exact_match_accuracy:.4f}")
    print(f"Micro F1 Score: {micro_f1:.4f}")
    print(f"Macro F1 Score: {macro_f1:.4f}")

    print("\nF1 Score by Label:")

    for index, label in enumerate(LABEL_COLUMNS):
        label_f1 = f1_score(
            y_test.iloc[:, index],
            predictions[:, index],
        )

        wandb.log({f"f1_{label}": label_f1})
        print(f"{label}: {label_f1:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    print(f"\nModel saved to: {MODEL_PATH}")

    model_artifact = wandb.Artifact(
        name="toxic-comment-model",
        type="model",
        description="TF-IDF and Logistic Regression multilabel toxicity classifier",
    )   

    model_artifact.add_file(MODEL_PATH)
    run.log_artifact(model_artifact)

    run.finish()


if __name__ == "__main__":
    main()