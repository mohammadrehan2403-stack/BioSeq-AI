"""BioSeq AI Advanced — standalone Gradio bioinformatics dashboard.
Educational prototype only. Outputs are not clinical diagnoses or validated
proof of biological function. BLAST submits sequences to NCBI over the internet.
"""
from __future__ import annotations

import json
import os
import tempfile
import urllib.request
import zipfile
from collections import Counter
from datetime import datetime
from itertools import product
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd
from Bio import Align, SeqIO
from Bio.Blast import NCBIWWW, NCBIXML
from Bio.Data import CodonTable
from Bio.Seq import Seq
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

APP_VERSION = "2.1.0"
DATA_URL = "https://archive.ics.uci.edu/static/public/67/molecular+biology+promoter+gene+sequences.zip"
DATA_DIR = Path("promoter_data")
ZIP_PATH = Path("promoter_dataset.zip")


def validate_dna(sequence, allow_n=False):
    if not isinstance(sequence, str) or not sequence.strip():
        raise ValueError("Please enter a DNA sequence.")
    cleaned = "".join(sequence.upper().split())
    allowed = set("ATGCN" if allow_n else "ATGC")
    invalid = sorted(set(cleaned) - allowed)
    if invalid:
        raise ValueError("Invalid DNA character(s): " + ", ".join(invalid))
    return cleaned


def load_dataset():
    data_file = DATA_DIR / "promoters.data"
    if not data_file.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not ZIP_PATH.exists():
            with urllib.request.urlopen(DATA_URL, timeout=45) as response:
                ZIP_PATH.write_bytes(response.read())
        with zipfile.ZipFile(ZIP_PATH) as archive:
            member = next((x for x in archive.namelist() if Path(x).name.lower() == "promoters.data"), None)
            if not member:
                raise RuntimeError("promoters.data was not found in the UCI archive.")
            archive.extract(member, DATA_DIR)
            extracted = DATA_DIR / member
            if extracted != data_file:
                data_file.write_bytes(extracted.read_bytes())
    rows = []
    with data_file.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            parts = line.strip().split(",")
            if len(parts) >= 3:
                label = parts[0].strip().strip('"')
                identifier = parts[1].strip().strip('"')
                sequence = "".join(parts[2].strip().upper().split()).strip('"')
                if label in {"+", "-"} and sequence and not (set(sequence) - set("ATGC")):
                    rows.append({"Class": label, "ID": identifier, "Sequence": sequence})
    dataset = pd.DataFrame(rows)
    if dataset.empty:
        raise RuntimeError("The UCI promoter dataset was downloaded but could not be parsed.")
    return dataset


def train_promoter_model(dataset):
    x_train, x_test, y_train, y_test = train_test_split(dataset["Sequence"], dataset["Class"], test_size=0.25, random_state=42, stratify=dataset["Class"])
    classifier = make_pipeline(CountVectorizer(analyzer="char", ngram_range=(2, 3)), LogisticRegression(max_iter=2000, random_state=42))
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_test)
    metrics = {
        "Dataset samples": int(len(dataset)), "Training samples": int(len(x_train)), "Test samples": int(len(x_test)),
        "Accuracy (%)": round(float(accuracy_score(y_test, predictions)) * 100, 2),
        "Macro precision (%)": round(float(precision_score(y_test, predictions, average="macro", zero_division=0)) * 100, 2),
        "Macro recall (%)": round(float(recall_score(y_test, predictions, average="macro", zero_division=0)) * 100, 2),
        "Macro F1-score (%)": round(float(f1_score(y_test, predictions, average="macro", zero_division=0)) * 100, 2),
        "Class counts": {str(k): int(v) for k, v in dataset["Class"].value_counts().items()},
        "Evaluation note": "One reproducible holdout split on a small dataset; not external validation.",
    }
    report = classification_report(y_test, predictions, labels=["+", "-"], target_names=["Promoter (+)", "Non-promoter (-)"], zero_division=0, output_dict=True)
    matrix = confusion_matrix(y_test, predictions, labels=["+", "-"])
    return classifier, metrics, pd.DataFrame(report).transpose(), matrix


try:
    df = load_dataset()
    model, MODEL_METRICS, MODEL_REPORT, MODEL_MATRIX = train_promoter_model(df)
    MODEL_ERROR = None
    print(f"Loaded {len(df)} promoter sequences; holdout accuracy {MODEL_METRICS['Accuracy (%)']}%")
except Exception as exc:
    df = pd.DataFrame(columns=["Class", "ID", "Sequence"])
    model = None
    MODEL_METRICS = {"Status": "Unavailable", "Reason": f"{type(exc).__name__}: {exc}"}
    MODEL_REPORT = pd.DataFrame()
    MODEL_MATRIX = None
    MODEL_ERROR = str(exc)
    print("Promoter model unavailable; the non-ML DNA tools will remain usable:", MODEL_ERROR)


def analyze_dna(sequence):
    dna = validate_dna(sequence)
    counts = Counter(dna)
    length = len(dna)
    gc = counts["G"] + counts["C"]
    at = counts["A"] + counts["T"]
    summary = {"Sequence length (bp)": length, "A count": counts["A"], "T count": counts["T"], "G count": counts["G"], "C count": counts["C"], "GC count": gc, "AT count": at, "GC content (%)": round(gc / length * 100, 2), "AT content (%)": round(at / length * 100, 2), "Sequence": dna}
    table = pd.DataFrame({"Nucleotide": list("ATGC"), "Count": [counts[b] for b in "ATGC"], "Percentage (%)": [round(counts[b] / length * 100, 2) for b in "ATGC"]})
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(table["Nucleotide"], table["Count"])
    ax.set_title("Nucleotide Composition"); ax.set_ylabel("Count"); ax.grid(axis="y", alpha=0.25); fig.tight_layout()
    return summary, table, fig


def reverse_complement(sequence):
    return str(Seq(validate_dna(sequence)).reverse_complement())


def translate_dna(sequence, frame=0, strand="Forward"):
    dna = Seq(validate_dna(sequence)); frame = int(frame)
    if frame not in (0, 1, 2): raise ValueError("Frame must be 0, 1, or 2.")
    if strand not in ("Forward", "Reverse"): raise ValueError("Choose Forward or Reverse strand.")
    if strand == "Reverse": dna = dna.reverse_complement()
    framed = dna[frame:]; usable = (len(framed) // 3) * 3; coding = framed[:usable]
    protein = str(coding.translate())
    rows = [{"Codon number": i // 3 + 1, "DNA codon": str(coding[i:i+3]), "Amino acid": str(coding[i:i+3].translate()), "Meaning": "Stop codon" if str(coding[i:i+3].translate()) == "*" else "Amino acid"} for i in range(0, len(coding), 3)]
    return {"Strand": strand, "Frame (0-based)": frame, "Coding DNA": str(coding), "Protein sequence": protein, "Complete codons": len(coding) // 3, "Ignored trailing bases": len(framed) - usable, "Stop codon count": protein.count("*")}, pd.DataFrame(rows)


def translate_six_frames(sequence):
    dna = validate_dna(sequence); rows = []
    for name, strand in [("Forward", Seq(dna)), ("Reverse", Seq(dna).reverse_complement())]:
        for frame in range(3):
            seq = strand[frame:]; usable = (len(seq) // 3) * 3; coding = seq[:usable]
            rows.append({"Reading frame": f"{name} frame {frame+1}", "Coding DNA": str(coding), "Protein sequence": str(coding.translate())})
    return pd.DataFrame(rows)


GENETIC_CODE = CodonTable.unambiguous_dna_by_name["Standard"]
AA_NAMES = {"A": ("Alanine", "Ala"), "R": ("Arginine", "Arg"), "N": ("Asparagine", "Asn"), "D": ("Aspartic acid", "Asp"), "C": ("Cysteine", "Cys"), "Q": ("Glutamine", "Gln"), "E": ("Glutamic acid", "Glu"), "G": ("Glycine", "Gly"), "H": ("Histidine", "His"), "I": ("Isoleucine", "Ile"), "L": ("Leucine", "Leu"), "K": ("Lysine", "Lys"), "M": ("Methionine", "Met"), "F": ("Phenylalanine", "Phe"), "P": ("Proline", "Pro"), "S": ("Serine", "Ser"), "T": ("Threonine", "Thr"), "W": ("Tryptophan", "Trp"), "Y": ("Tyrosine", "Tyr"), "V": ("Valine", "Val")}
CODON_ROWS = []
for triplet in product("TCAG", repeat=3):
    codon = "".join(triplet)
    if codon in GENETIC_CODE.stop_codons: aa, short, one, kind = "Stop", "Ter", "*", "Stop"
    else:
        one = GENETIC_CODE.forward_table[codon]; aa, short = AA_NAMES[one]; kind = "Start" if codon == "ATG" else "Standard"
    CODON_ROWS.append({"Codon": codon, "Amino Acid": aa, "Three Letter": short, "One Letter": one, "Type": kind})
CODON_TABLE = pd.DataFrame(CODON_ROWS).sort_values("Codon").reset_index(drop=True)


def search_codon(query):
    q = str(query or "").strip().lower()
    if not q: return CODON_TABLE.copy()
    mask = CODON_TABLE.apply(lambda row: row.astype(str).str.lower().str.contains(q, regex=False).any(), axis=1)
    return CODON_TABLE[mask].reset_index(drop=True)


def compare_dna_sequences(reference, query):
    ref, qry = validate_dna(reference), validate_dna(query)
    aligner = Align.PairwiseAligner(); aligner.mode = "global"; aligner.match_score = 2; aligner.mismatch_score = -1; aligner.open_gap_score = -3; aligner.extend_gap_score = -0.5
    alignment = aligner.align(ref, qry)[0]; coordinates = alignment.coordinates
    ar_parts, aq_parts, differences = [], [], []; matches = ref_pos = qry_pos = 0
    for i in range(coordinates.shape[1] - 1):
        r0, r1 = int(coordinates[0, i]), int(coordinates[0, i+1]); q0, q1 = int(coordinates[1, i]), int(coordinates[1, i+1])
        rc, qc = ref[r0:r1], qry[q0:q1]; rs, qs = r1-r0, q1-q0
        if rs > 0 and qs > 0:
            for rb, qb in zip(rc, qc):
                ar_parts.append(rb); aq_parts.append(qb); ref_pos += 1; qry_pos += 1
                if rb == qb: matches += 1
                else: differences.append({"Reference position": ref_pos, "Query position": qry_pos, "Reference base": rb, "Query base": qb, "Variation": "Substitution"})
        elif rs > 0:
            for rb in rc:
                ar_parts.append(rb); aq_parts.append("-"); ref_pos += 1
                differences.append({"Reference position": ref_pos, "Query position": "-", "Reference base": rb, "Query base": "-", "Variation": "Deletion"})
        elif qs > 0:
            for qb in qc:
                ar_parts.append("-"); aq_parts.append(qb); qry_pos += 1
                differences.append({"Reference position": "-", "Query position": qry_pos, "Reference base": "-", "Query base": qb, "Variation": "Insertion"})
    ar, aq = "".join(ar_parts), "".join(aq_parts); length = len(ar)
    summary = {"Reference length": len(ref), "Query length": len(qry), "Alignment length": length, "Matches": matches, "Substitutions": sum(x["Variation"] == "Substitution" for x in differences), "Insertions": sum(x["Variation"] == "Insertion" for x in differences), "Deletions": sum(x["Variation"] == "Deletion" for x in differences), "Sequence identity (%)": round(matches / length * 100, 2) if length else 0, "Alignment score": float(alignment.score), "Aligned reference": ar, "Aligned query": aq}
    return summary, pd.DataFrame(differences, columns=["Reference position", "Query position", "Reference base", "Query base", "Variation"])


def detect_orfs(sequence, min_length=1):
    dna = validate_dna(sequence); min_length = int(min_length)
    if min_length < 1: raise ValueError("Minimum ORF length must be at least 1 codon.")
    results = []
    for name, strand in [("Forward", dna), ("Reverse", str(Seq(dna).reverse_complement()))]:
        for frame in range(3):
            starts = []
            for pos in range(frame, len(strand)-2, 3):
                codon = strand[pos:pos+3]
                if codon == "ATG": starts.append(pos)
                elif codon in ("TAA", "TAG", "TGA"):
                    for start in starts:
                        end = pos+3
                        if (end-start)//3 >= min_length:
                            segment = strand[start:end]
                            results.append({"Strand": name, "Frame": frame+1, "Start position (1-based)": start+1 if name == "Forward" else len(dna)-end+1, "End position (1-based)": end if name == "Forward" else len(dna)-start, "Length (bp)": end-start, "Start codon": "ATG", "Stop codon": codon, "DNA sequence": segment, "Protein sequence": str(Seq(segment).translate())})
                    starts = []
    return pd.DataFrame(results, columns=["Strand", "Frame", "Start position (1-based)", "End position (1-based)", "Length (bp)", "Start codon", "Stop codon", "DNA sequence", "Protein sequence"])


def search_motif(sequence, motif):
    dna, target = validate_dna(sequence), validate_dna(motif); hits=[]; start=0
    while True:
        pos=dna.find(target,start)
        if pos < 0: break
        hits.append({"Motif": target, "Start position (1-based)": pos+1, "End position (1-based)": pos+len(target), "Matched sequence": dna[pos:pos+len(target)]}); start=pos+1
    return pd.DataFrame(hits, columns=["Motif", "Start position (1-based)", "End position (1-based)", "Matched sequence"])


def gc_sliding_window(sequence, window_size=10, step_size=1):
    dna=validate_dna(sequence); window_size=int(window_size); step_size=int(step_size)
    if window_size<1 or step_size<1: raise ValueError("Window size and step size must be positive integers.")
    if window_size>len(dna): raise ValueError("Window size cannot exceed sequence length.")
    rows=[]
    for start in range(0,len(dna)-window_size+1,step_size):
        window=dna[start:start+window_size]; gc=window.count("G")+window.count("C")
        rows.append({"Start position":start+1,"End position":start+window_size,"DNA window":window,"GC count":gc,"GC percentage":round(gc/window_size*100,2)})
    return pd.DataFrame(rows)


def kmer_analysis(sequence, k=3, top_n=15):
    dna=validate_dna(sequence); k=int(k); top_n=int(top_n)
    if k<1 or k>len(dna): raise ValueError("K-mer length must be between 1 and sequence length.")
    if top_n<1: raise ValueError("Top results must be at least 1.")
    counts=Counter(dna[i:i+k] for i in range(len(dna)-k+1))
    return pd.DataFrame(counts.most_common(min(top_n,len(counts))),columns=["K-mer","Frequency"])


def dna_quality_report(sequence):
    dna=validate_dna(sequence,allow_n=True); counts={b:dna.count(b) for b in "ATGCN"}; length=len(dna); known=length-counts["N"]
    return pd.DataFrame([
        {"Metric":"Sequence length (bp)","Value":length}, {"Metric":"A count","Value":counts["A"]}, {"Metric":"T count","Value":counts["T"]}, {"Metric":"G count","Value":counts["G"]}, {"Metric":"C count","Value":counts["C"]}, {"Metric":"Ambiguous N count","Value":counts["N"]},
        {"Metric":"GC content among known bases (%)","Value":round((counts["G"]+counts["C"])/known*100,2) if known else 0}, {"Metric":"Ambiguous bases (%)","Value":round(counts["N"]/length*100,2)}, {"Metric":"Only A/T/G/C/N symbols","Value":True}])


def analyze_protein(protein_sequence):
    protein=(protein_sequence or "").strip().upper()
    if protein.endswith("*"): protein=protein[:-1]
    if not protein: raise ValueError("Protein sequence cannot be empty.")
    if set(protein)-set("ACDEFGHIKLMNPQRSTVWY"): raise ValueError("Use only the 20 standard amino-acid letters; a terminal * is optional.")
    analysis=ProteinAnalysis(protein); counts=analysis.count_amino_acids(); total=len(protein)
    table=pd.DataFrame([{"Amino acid":aa,"Count":n,"Percentage (%)":round(n/total*100,2)} for aa,n in sorted(counts.items()) if n])
    summary={"Protein sequence":protein,"Length (amino acids)":total,"Estimated molecular weight (Da)":round(analysis.molecular_weight(),2),"Estimated isoelectric point (pI)":round(analysis.isoelectric_point(),3),"Instability index":round(analysis.instability_index(),2),"GRAVY hydropathicity":round(analysis.gravy(),3)}
    return summary,table


def blast_dna_search(sequence,max_results=10):
    dna=validate_dna(sequence); max_results=int(max_results)
    if len(dna)<50: raise ValueError("BLAST search requires at least 50 bases in this demo.")
    if not 1<=max_results<=20: raise ValueError("Maximum results must be between 1 and 20.")
    handle=None
    try:
        handle=NCBIWWW.qblast(program="blastn",database="nt",sequence=dna,hitlist_size=max_results,expect=10.0,format_type="XML")
        record=NCBIXML.read(handle)
    except Exception as exc: raise RuntimeError("NCBI BLAST failed or timed out. Try again later. Details: "+str(exc)) from exc
    finally:
        if handle is not None: handle.close()
    rows=[]
    for alignment in record.alignments:
        if not alignment.hsps: continue
        hsp=min(alignment.hsps,key=lambda item:item.expect)
        rows.append({"Accession ID":alignment.accession,"Description":alignment.hit_def,"Alignment length":hsp.align_length,"Identical bases":hsp.identities,"Identity (%)":round(hsp.identities/hsp.align_length*100,2) if hsp.align_length else 0,"Query coverage (%)":round((hsp.query_end-hsp.query_start+1)/len(dna)*100,2),"E-value":hsp.expect,"Bit score":round(hsp.bits,2)})
    return pd.DataFrame(rows)


def promoter_prediction(sequence):
    dna=validate_dna(sequence)
    if len(dna)!=57: raise ValueError("Promoter model expects exactly 57 bases.")
    if model is None: raise RuntimeError("Promoter model unavailable because dataset load failed: "+str(MODEL_ERROR))
    pred=str(model.predict([dna])[0]); probs={}
    if hasattr(model,"predict_proba"):
        probs={str(label):round(float(value)*100,2) for label,value in zip(model.classes_,model.predict_proba([dna])[0])}
    return {"Input length":len(dna),"Predicted class":"Promoter (+)" if pred=="+" else "Non-promoter (-)","Predicted label":pred,"Dataset":"UCI Molecular Biology Promoter Sequences","Dataset size":int(len(df)),"Held-out test accuracy (%)":MODEL_METRICS.get("Accuracy (%)","Unavailable"),"Model probabilities (%)":probs,"Limitation":"Educational prototype, not proof of biological function or for clinical use."},pd.DataFrame([{"Class":k,"Probability (%)":v} for k,v in probs.items()])


def safe_call(fn,*args):
    try: return fn(*args)
    except Exception as exc: return {"error":f"{type(exc).__name__}: {exc}"}


def cb_analyze(seq):
    try: return *analyze_dna(seq),
    except Exception as exc: return {"error":str(exc)},pd.DataFrame(),None

def cb_reverse(seq):
    try:return reverse_complement(seq)
    except Exception as exc:return "Error: "+str(exc)

def cb_translate(seq,frame,strand):
    try:return translate_dna(seq,frame,strand)
    except Exception as exc:return {"error":str(exc)},pd.DataFrame()

def cb_six(seq):
    try:return translate_six_frames(seq)
    except Exception as exc:return pd.DataFrame([{"Error":str(exc)}])

def cb_codon(query): return search_codon(query)

def cb_compare(ref,query):
    try:
        summary,table=compare_dna_sequences(ref,query)
        aligned=f"Reference: {summary['Aligned reference']}\nQuery:     {summary['Aligned query']}\n\nIdentity: {summary['Sequence identity (%)']}%"
        return summary,table,aligned
    except Exception as exc:return {"error":str(exc)},pd.DataFrame(),""

def cb_orf(seq,n):
    try:return detect_orfs(seq,n),"ORF search complete. Coordinates are reported 1-based."
    except Exception as exc:return pd.DataFrame(),str(exc)

def cb_motif(seq,motif):
    try:
        table=search_motif(seq,motif);return table,f"Found {len(table)} occurrence(s); positions are 1-based."
    except Exception as exc:return pd.DataFrame(),str(exc)

def cb_gc(seq,w,s):
    try:
        table=gc_sliding_window(seq,w,s); fig,ax=plt.subplots(figsize=(7,3.5));ax.plot(table["Start position"],table["GC percentage"],marker="o");ax.set_title("GC content by sliding window");ax.set_xlabel("Window start position");ax.set_ylabel("GC (%)");ax.set_ylim(0,100);ax.grid(True,alpha=.3);fig.tight_layout();return table,fig,"Analysis complete."
    except Exception as exc:return pd.DataFrame(),None,str(exc)

def cb_kmer(seq,k,top):
    try:
        table=kmer_analysis(seq,k,top);fig,ax=plt.subplots(figsize=(7,3.5));ax.bar(table["K-mer"],table["Frequency"]);ax.set_title("Most frequent k-mers");ax.set_ylabel("Frequency");ax.tick_params(axis="x",rotation=45);fig.tight_layout();return table,fig,"Analysis complete."
    except Exception as exc:return pd.DataFrame(),None,str(exc)

def cb_quality(seq):
    try:return dna_quality_report(seq),"Sequence quality report ready. N denotes an ambiguous base."
    except Exception as exc:return pd.DataFrame(),str(exc)

def cb_protein(seq):
    try:return analyze_protein(seq)
    except Exception as exc:return {"error":str(exc)},pd.DataFrame()

def cb_promoter(seq):
    try:return promoter_prediction(seq)
    except Exception as exc:return {"error":str(exc)},pd.DataFrame()

def cb_performance():
    if MODEL_MATRIX is None:return MODEL_METRICS,MODEL_REPORT,None
    fig,ax=plt.subplots(figsize=(5,4));ax.imshow(MODEL_MATRIX);ax.set_title("Confusion matrix (holdout split)");ax.set_xlabel("Predicted");ax.set_ylabel("Actual");ax.set_xticks([0,1],["Promoter (+)","Non-promoter (-)"]);ax.set_yticks([0,1],["Promoter (+)","Non-promoter (-)"])
    for i in range(2):
        for j in range(2):ax.text(j,i,str(MODEL_MATRIX[i,j]),ha="center",va="center")
    fig.tight_layout();return MODEL_METRICS,MODEL_REPORT,fig

def cb_blast(seq,n):
    try:
        table=blast_dna_search(seq,n);return table,f"NCBI returned {len(table)} match(es). A match alone does not confirm biological function."
    except Exception as exc:return pd.DataFrame(),"BLAST error: "+str(exc)

def cb_fasta(path):
    if not path:return pd.DataFrame(),"Choose a FASTA file first."
    try:
        rows=[]
        for record in SeqIO.parse(str(path),"fasta"):
            seq=str(record.seq).upper(); invalid=sorted(set(seq)-set("ATGCN"));length=len(seq);known=length-seq.count("N");gc=seq.count("G")+seq.count("C")
            rows.append({"ID":record.id,"Description":record.description,"Length (bp)":length,"GC content among known bases (%)":round(gc/known*100,2) if known else 0,"N count":seq.count("N"),"Status":"Empty sequence" if not seq else ("Invalid symbols: "+", ".join(invalid) if invalid else "OK")})
        return pd.DataFrame(rows),f"Processed {len(rows)} FASTA record(s)."
    except Exception as exc:return pd.DataFrame(),f"FASTA error: {exc}"

def cb_report(seq,ref,query,name):
    try:
        dna=validate_dna(seq,allow_n=True)
        report={"project":"BioSeq AI Advanced","version":APP_VERSION,"generated_at":datetime.now().isoformat(timespec="seconds"),"sequence_name":str(name or "DNA_Sequence"),"dna_quality":dna_quality_report(dna).to_dict(orient="records")}
        if "N" not in dna:report["dna_analysis"]=analyze_dna(dna)[0]
        if str(ref or "").strip() and str(query or "").strip():
            try:report["sequence_comparison"]=compare_dna_sequences(ref,query)[0]
            except Exception as exc:report["comparison_error"]=str(exc)
        fd,path=tempfile.mkstemp(prefix="bioseq_ai_report_",suffix=".json");os.close(fd)
        with open(path,"w",encoding="utf-8") as stream:json.dump(report,stream,indent=2,ensure_ascii=False,default=str)
        return path,"JSON report ready to download."
    except Exception as exc:return None,"Report error: "+str(exc)


with gr.Blocks(title="BioSeq AI — Advanced Dashboard", theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# 🧬 BioSeq AI — Advanced Dashboard\nVersion {APP_VERSION} · DNA analysis, translation, alignment and educational ML classification")
    gr.Markdown("Educational prototype only. Results require scientific interpretation. Not for medical diagnosis. NCBI BLAST sends the input sequence to an external service.")
    with gr.Tabs():
        with gr.Tab("🏠 Overview"):
            if model is None: gr.Markdown("### Promoter classifier unavailable\n"+str(MODEL_ERROR))
            else: gr.Markdown(f"### Promoter model ready\n- Dataset: {len(df)} records from UCI\n- Held-out accuracy on one split: {MODEL_METRICS.get('Accuracy (%)')}%\n- Tools: DNA statistics, reverse complement, protein and six-frame translation, codon explorer, sequence comparison, ORFs, motif search, GC windows, k-mers, quality report, protein properties, BLAST, FASTA batch and JSON export.")
        with gr.Tab("🧬 DNA Analyzer"):
            seq=gr.Textbox(label="DNA sequence",value="ATGCGCTA",lines=3);btn=gr.Button("Analyze DNA",variant="primary");summary=gr.JSON(label="DNA analysis");table=gr.Dataframe(label="Nucleotide counts and percentages");plot=gr.Plot(label="Composition chart");btn.click(cb_analyze,[seq],[summary,table,plot])
        with gr.Tab("🔄 Reverse Complement"):
            seq=gr.Textbox(label="DNA sequence",value="ATGCGTAC",lines=2);btn=gr.Button("Generate reverse complement",variant="primary");out=gr.Textbox(label="Result",lines=3);btn.click(cb_reverse,[seq],[out])
        with gr.Tab("🧪 Protein Translation"):
            seq=gr.Textbox(label="DNA sequence",value="ATGGCTTTTTAA",lines=3)
            with gr.Row(): frame=gr.Dropdown([0,1,2],value=0,label="Reading frame (0-based)");strand=gr.Dropdown(["Forward","Reverse"],value="Forward",label="Strand")
            btn=gr.Button("Translate selected frame",variant="primary");summary=gr.JSON(label="Translation details");table=gr.Dataframe(label="Codon-by-codon translation");btn.click(cb_translate,[seq,frame,strand],[summary,table])
            gr.Markdown("#### Six-frame translation");seq6=gr.Textbox(label="DNA sequence for all six frames",value="ATGGCTTTTTAA",lines=2);btn6=gr.Button("Translate all six frames");out6=gr.Dataframe(label="Six-frame protein sequences");btn6.click(cb_six,[seq6],[out6])
        with gr.Tab("🧬 Codon Explorer"):
            query=gr.Textbox(label="Search by codon or amino-acid name/code (blank displays all 64)",value="Methionine");btn=gr.Button("Search codons");cod=gr.Dataframe(value=CODON_TABLE,label="Genetic code reference");btn.click(cb_codon,[query],[cod])
        with gr.Tab("🔬 Sequence Comparison"):
            ref=gr.Textbox(label="Reference DNA",value="ATGCGTAC",lines=2);query=gr.Textbox(label="Query DNA",value="ATGCTTAC",lines=2);btn=gr.Button("Align and detect differences",variant="primary");summary=gr.JSON(label="Alignment summary");diff=gr.Dataframe(label="Substitutions, insertions and deletions");aligned=gr.Textbox(label="Aligned sequences",lines=5);btn.click(cb_compare,[ref,query],[summary,diff,aligned])
        with gr.Tab("🧬 ORF Finder"):
            seq=gr.Textbox(label="DNA sequence",value="CCCATGAAATAGGGG",lines=3);minimum=gr.Number(label="Minimum ORF length (codons, including stop)",value=1,precision=0);btn=gr.Button("Find open reading frames",variant="primary");out=gr.Dataframe(label="ORF results");note=gr.Textbox(label="Status");btn.click(cb_orf,[seq,minimum],[out,note])
        with gr.Tab("🔎 Motif Search"):
            seq=gr.Textbox(label="DNA sequence",value="ATGCGTACATG",lines=2);motif=gr.Textbox(label="DNA motif",value="ATG");btn=gr.Button("Search motif",variant="primary");out=gr.Dataframe(label="Matching positions");note=gr.Textbox(label="Status");btn.click(cb_motif,[seq,motif],[out,note])
        with gr.Tab("📈 GC Sliding Window"):
            seq=gr.Textbox(label="DNA sequence",value="ATGCGCGCATATATGC",lines=3)
            with gr.Row():window=gr.Number(label="Window size (bases)",value=6,precision=0);step=gr.Number(label="Step size",value=3,precision=0)
            btn=gr.Button("Analyze GC by window",variant="primary");out=gr.Dataframe(label="Window-by-window GC content");plot=gr.Plot(label="GC trend");note=gr.Textbox(label="Status");btn.click(cb_gc,[seq,window,step],[out,plot,note])
        with gr.Tab("📊 K-mer Analysis"):
            seq=gr.Textbox(label="DNA sequence",value="ATGCGTACATG",lines=2)
            with gr.Row():k=gr.Number(label="K-mer length",value=3,precision=0);top=gr.Number(label="Top results",value=15,precision=0)
            btn=gr.Button("Count k-mers",variant="primary");out=gr.Dataframe(label="K-mer frequency");plot=gr.Plot(label="Most frequent k-mers");note=gr.Textbox(label="Status");btn.click(cb_kmer,[seq,k,top],[out,plot,note])
        with gr.Tab("🧾 DNA Quality"):
            seq=gr.Textbox(label="DNA sequence (N allowed for ambiguous bases)",value="ATGCGCATNNATGC",lines=2);btn=gr.Button("Check sequence quality",variant="primary");out=gr.Dataframe(label="Quality report");note=gr.Textbox(label="Status");btn.click(cb_quality,[seq],[out,note])
        with gr.Tab("🧪 Protein Analysis"):
            prot=gr.Textbox(label="Protein sequence",value="MKWVTF",lines=2);btn=gr.Button("Analyze protein",variant="primary");summary=gr.JSON(label="Protein properties");table=gr.Dataframe(label="Amino-acid composition");btn.click(cb_protein,[prot],[summary,table])
        with gr.Tab("🤖 AI Promoter Classifier"):
            seq=gr.Textbox(label="DNA sequence (exactly 57 bases)",placeholder="Enter 57 bases using A/T/G/C",lines=3);btn=gr.Button("Predict promoter class",variant="primary");summary=gr.JSON(label="Prediction");table=gr.Dataframe(label="Class probabilities");btn.click(cb_promoter,[seq],[summary,table])
        with gr.Tab("📉 Model Performance"):
            btn=gr.Button("Show holdout metrics",variant="primary");summary=gr.JSON(label="Evaluation metrics");table=gr.Dataframe(label="Precision, recall and F1 report");plot=gr.Plot(label="Confusion matrix");btn.click(cb_performance,[],[summary,table,plot])
        with gr.Tab("🌐 NCBI BLAST"):
            gr.Markdown("BLAST uses NCBI's remote service and may take several minutes. Minimum 50 DNA bases; external connectivity required.");seq=gr.Textbox(label="DNA sequence (minimum 50 bases)",lines=4);n=gr.Number(label="Maximum results (1–20)",value=10,precision=0);btn=gr.Button("Search NCBI BLAST");out=gr.Dataframe(label="BLAST matches");note=gr.Textbox(label="Status / interpretation",lines=4);btn.click(cb_blast,[seq,n],[out,note])
        with gr.Tab("📂 FASTA Batch"):
            gr.Markdown("Upload a FASTA file (.fa/.fasta); each record is summarized.");file=gr.File(label="FASTA file",file_count="single",type="filepath");btn=gr.Button("Analyze FASTA file");out=gr.Dataframe(label="FASTA records");note=gr.Textbox(label="Status");btn.click(cb_fasta,[file],[out,note])
        with gr.Tab("📥 Export JSON Report"):
            seq=gr.Textbox(label="DNA sequence for report",value="ATGCGCTA",lines=2);name=gr.Textbox(label="Sequence name",value="BioSeqAI_Sequence")
            with gr.Row():ref=gr.Textbox(label="Optional reference DNA",lines=2);query=gr.Textbox(label="Optional query DNA",lines=2)
            btn=gr.Button("Generate downloadable JSON report",variant="primary");file=gr.File(label="Download report");note=gr.Textbox(label="Status");btn.click(cb_report,[seq,ref,query,name],[file,note])
    gr.Markdown("---\n**Limitations:** Small promoter dataset and a single holdout split; BLAST requires external connectivity. Sequence analyses are educational and are not clinical variant analysis.")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    demo.launch(server_name="0.0.0.0", server_port=port, show_error=True)
