# BioSeq AI — DNA Sequence Classification Using Machine Learning

BioSeq AI is an educational machine-learning prototype that classifies a DNA sequence as a promoter (`+`) or non-promoter (`-`) using the UCI Molecular Biology Promoter dataset.

## Features

- Downloads and parses the dataset on first run.
- Converts DNA sequences into character 2-mer and 3-mer features.
- Trains a Logistic Regression classifier in a scikit-learn pipeline.
- Splits raw sequences into training and test sets before fitting the vectorizer.
- Reports test-set accuracy, classification report, and confusion matrix in the terminal.
- Provides a Gradio web interface for predictions and nucleotide-composition statistics.
- Rejects empty sequences and characters other than A, T, G, and C.

## Requirements

- Python 3.10 or newer recommended
- Internet connection on first run to download the dataset

## Run locally

```bash
python -m venv .venv
```

Activate the environment:

- Windows PowerShell: `.venv\Scripts\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the app:

```bash
python bioseq_ai.py
```

Open the local URL printed in the terminal. To create a temporary public share link for a demo, change the final line to `demo.launch(share=True)`; share links are temporary.

## Dataset

UCI Machine Learning Repository, **Molecular Biology (Promoter Gene Sequences)**:
https://archive.ics.uci.edu/dataset/67/molecular+biology+promoter+gene+sequences

The application uses `+` as promoter and `-` as non-promoter for this dataset.

## Method

1. Load DNA sequences and their dataset labels.
2. Split sequences into training and held-out test sets (75% / 25%, stratified).
3. Use `CountVectorizer(analyzer="char", ngram_range=(2, 3))` to encode character 2-mers and 3-mers.
4. Train Logistic Regression.
5. Evaluate on the held-out test set and display results.
6. Use the trained pipeline for new sequence predictions.

## Limitations and responsible use

- The dataset is small (106 sequences in the version used for this project).
- A single train/test split can give an unstable estimate of performance.
- Test-set accuracy is an experimental result, not a guarantee for unseen real-world sequences.
- This prototype is for learning and demonstration only. It is not a medical diagnostic tool and must not be used for clinical decisions.

## Project structure

```text
BioSeq-AI/
├── bioseq_ai.py
├── requirements.txt
├── README.md
└── .gitignore
```

## License

Educational project. Check the dataset repository's terms and attribution requirements before redistribution.
## 🧬 BioSeq AI — Advanced Bioinformatics Dashboard

An interactive DNA sequence analysis and machine-learning project built using Python and Gradio.

### Live Demo

[Open BioSeq AI](https://bioseq-ai-final-demo.onrender.com/)

### Source Code

[View the GitHub Repository](https://github.com/mohammadrehan2403-stack/BioSeq-AI)

### Key Features

* DNA sequence analysis and nucleotide composition
* Reverse complement generation
* Protein translation and six-frame translation
* Codon exploration
* DNA sequence alignment and mutation comparison
* Open Reading Frame (ORF) detection
* DNA motif and K-mer analysis
* GC-content sliding-window visualization
* Protein sequence analysis
* AI-based promoter classification
* NCBI BLAST search
* FASTA batch analysis
* Downloadable JSON reports

### Technology Stack

Python, Gradio, Biopython, Pandas, Matplotlib and Scikit-learn.

### Disclaimer

This is an educational bioinformatics project. Its outputs are not medical diagnoses or validated clinical conclusions.
