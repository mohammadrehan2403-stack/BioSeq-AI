"""BioSeq AI: DNA sequence classification demo using the UCI promoter dataset.

This is an educational prototype, not a clinical or medical diagnostic tool.
"""
from collections import Counter
from pathlib import Path
import urllib.request
import zipfile

import gradio as gr
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_URL = (
    "https://archive.ics.uci.edu/static/public/67/"
    "molecular+biology+promoter+gene+sequences.zip"
)
DATA_DIR = Path("promoter_data")
ZIP_PATH = Path("promoter_dataset.zip")


def load_dataset():
    """Download and parse the UCI Molecular Biology Promoter dataset."""
    data_file = DATA_DIR / "promoters.data"

    if not data_file.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not ZIP_PATH.exists():
            print("Downloading UCI promoter dataset...")
            urllib.request.urlretrieve(DATA_URL, ZIP_PATH)
        with zipfile.ZipFile(ZIP_PATH, "r") as archive:
            archive.extractall(DATA_DIR)

    rows = []
    with data_file.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.strip().split(",")
            if len(parts) >= 3:
                label = parts[0].strip()
                gene_name = parts[1].strip()
                sequence = parts[2].strip().lower()
                if label in {"+", "-"} and sequence and all(
                    base in "atgc" for base in sequence
                ):
                    rows.append({
                        "label": label,
                        "gene_name": gene_name,
                        "sequence": sequence,
                    })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(
            "The dataset could not be parsed. Check the downloaded dataset format."
        )
    return df


def train_model(df):
    """Split raw sequences first, then fit vectorizer and classifier together."""
    X_train, X_test, y_train, y_test = train_test_split(
        df["sequence"],
        df["label"],
        test_size=0.25,
        random_state=42,
        stratify=df["label"],
    )
    pipeline = make_pipeline(
        CountVectorizer(analyzer="char", ngram_range=(2, 3)),
        LogisticRegression(max_iter=2000),
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(
        y_test, predictions, zero_division=0
    )
    matrix = confusion_matrix(y_test, predictions, labels=["+", "-"])
    return pipeline, accuracy, report, matrix, len(X_train), len(X_test)


def predict_dna(sequence):
    sequence = (sequence or "").strip().upper()

    if not sequence:
        return "Please enter a DNA sequence.", ""

    if any(base not in "ATGC" for base in sequence):
        return "Invalid input. Use only A, T, G and C.", ""

    if len(sequence) < 3:
        return "Sequence too short. Enter at least 3 bases.", ""

    prediction = model.predict([sequence])[0]
    counts = Counter(sequence)
    total = len(sequence)
    stats = "\n".join([
        f"Sequence length: {total} bases",
        f"A: {counts['A']} ({counts['A'] / total * 100:.1f}%)",
        f"T: {counts['T']} ({counts['T'] / total * 100:.1f}%)",
        f"G: {counts['G']} ({counts['G'] / total * 100:.1f}%)",
        f"C: {counts['C']} ({counts['C'] / total * 100:.1f}%)",
    ])

    # In the UCI promoter dataset, + denotes promoter and - denotes non-promoter.
    label_meaning = "promoter" if prediction == "+" else "non-promoter"
    result = (
        f"Predicted class: {prediction} ({label_meaning})\n\n"
        "This is an experimental prediction from a small educational dataset. "
        "It is not a medical diagnosis or a validated biological conclusion."
    )
    return result, stats


df = load_dataset()
model, accuracy, report, matrix, train_count, test_count = train_model(df)

print("BioSeq AI model ready.")
print(f"Dataset sequences: {len(df)}")
print(f"Training sequences: {train_count}")
print(f"Testing sequences: {test_count}")
print(f"Test-set accuracy: {accuracy * 100:.2f}%")
print("\nClassification report:\n", report)
print("\nConfusion matrix (labels +, -):\n", matrix)

with gr.Blocks(title="BioSeq AI") as demo:
    gr.Markdown("# 🧬 BioSeq AI")
    gr.Markdown("### DNA Sequence Classification Using Machine Learning")
    gr.Markdown(
        "Enter a DNA sequence using only A, T, G and C. "
        "The model is trained on the UCI Molecular Biology Promoter dataset."
    )
    with gr.Row():
        gr.Markdown(f"**Dataset size:** {len(df)} sequences")
        gr.Markdown(f"**Held-out test accuracy:** {accuracy * 100:.2f}%")
    dna_input = gr.Textbox(
        label="DNA sequence",
        placeholder="Example: ATGCGTACGTAGCTAG",
        lines=3,
    )
    classify_button = gr.Button("🔬 Classify DNA Sequence", variant="primary")
    prediction_output = gr.Textbox(label="Classification result", lines=3)
    stats_output = gr.Textbox(label="DNA sequence statistics", lines=5)
    classify_button.click(
        fn=predict_dna,
        inputs=dna_input,
        outputs=[prediction_output, stats_output],
    )
    gr.Markdown(
        "**Limitations:** This is a small-dataset educational prototype. "
        "Performance on this split does not guarantee performance on new data. "
        "Do not use for medical diagnosis or clinical decisions."
    )

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    demo.launch(server_name="0.0.0.0", server_port=port)

