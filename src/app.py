import os
import gradio as gr
from huggingface_hub import hf_hub_download
import pandas as pd
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

MONITORING_REPO = config["monitoring"]["monitoring_dataset_id"]
METRICS_PATH_REMOTE = "metrics/metrics_history.csv"
METRICS_LOCAL = "metrics_history.csv"


def load_metrics_data():
    token = os.getenv("HF_TOKEN")
    df = None
    source = ""

    if token:
        try:
	        # Scarica sempre l'ultima versione dal Dataset Hub
            path = hf_hub_download(
                repo_id=MONITORING_REPO,
                filename=METRICS_PATH_REMOTE,
                repo_type="dataset",
                token=token,
            )
            df = pd.read_csv(path)
            source = f"HF Dataset ({MONITORING_REPO}/metrics)"
        except Exception:
            df = None

    if df is None or df.empty:
        if os.path.exists(METRICS_LOCAL):
            try:
                df = pd.read_csv(METRICS_LOCAL)
                source = "Cache Locale"
            except Exception:
                df = None

    # Fallback se il dataset è ancora vuoto o non configurato
    if df is None or df.empty:
        empty_df = pd.DataFrame(
            columns=["timestamp", "batch_id", "run_id", "f1_score", "neg_pct", "neu_pct", "pos_pct", "event_type"]
        )
        return empty_df, "Nessun dato disponibile."

    latest_f1 = df["f1_score"].iloc[-1]
    status = f"Ultima F1: {latest_f1:.4f} | Eventi: {len(df)} | Sorgente: {source}"
    return df, status


with gr.Blocks(title="MLOps Sentiment Dashboard") as demo:
    gr.Markdown("# 📊 Osservabilità MLOps: Twitter-RoBERTa Sentiment Pipeline")
    gr.Markdown(
        "Monitoraggio streaming temporale: degradamento metrica $F_1$, "
        "rilevamento drift sul sentiment e tracciamento ripristino post-retrain."
    )

    status_box = gr.Textbox(label="Stato Pipeline e Sorgente Telemetria", interactive=False)

    with gr.Row():
        refresh_btn = gr.Button("🔄 Aggiorna Metriche e Grafici", variant="primary")

    gr.Markdown("### 1. Andamento Temporale Performance (F1-Score)")
    f1_plot = gr.LinePlot(
        value=load_metrics_data()[0],
        x="batch_id",
        y="f1_score",
        color="event_type",
        title="F1 Score per Batch di Produzione",
        y_lim=[0.0, 1.0],
        tooltip=["batch_id", "run_id", "f1_score", "event_type"],
        height=320,
    )

    gr.Markdown("### 2. Distribuzione del Sentiment Rilevato nel Flusso (%)")
    sentiment_plot = gr.LinePlot(
        value=load_metrics_data()[0],
        x="batch_id",
        y="pos_pct",
        color="event_type",
        title="Percentuale Tweet Positivi Rilevati per Batch (Indice di Drift)",
        y_lim=[0.0, 100.0],
        tooltip=["batch_id", "neg_pct", "neu_pct", "pos_pct"],
        height=320,
    )

    gr.Markdown("### 3. Registro Eventi e Data Lineage")
    table = gr.DataFrame(value=load_metrics_data()[0], interactive=False)

    def update_view():
        data, status = load_metrics_data()
        return data, data, data, status

    refresh_btn.click(fn=update_view, outputs=[f1_plot, sentiment_plot, table, status_box])
    demo.load(fn=update_view, outputs=[f1_plot, sentiment_plot, table, status_box])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
