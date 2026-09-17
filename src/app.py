import os
import gradio as gr
import pandas as pd
import yaml
from huggingface_hub import hf_hub_download
from src.metrics_logger import METRICS_FILENAME

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

REPO_ID = config["monitoring"]["metrics_dataset_id"]


def load_metrics_data():
    token = os.getenv("HF_TOKEN")
    try:
        # Scarica sempre l'ultima versione dal Dataset Hub
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=METRICS_FILENAME,
            repo_type="dataset",
            token=token,
        )
        df = pd.read_csv(path)
        latest_f1 = df["f1_score"].iloc[-1]
        status = f"Ultima F1: {latest_f1:.4f} | Eventi totali: {len(df)} (Sorgente: HF Dataset)"
        return df, status
    except Exception as e:
        # Fallback se il dataset è ancora vuoto o non configurato
        empty_df = pd.DataFrame(
            columns=["timestamp", "batch_id", "model_id", "f1_score", "is_drifted", "event_type"]
        )
        return empty_df, f"Nessun dato remoto disponibile ({e})"


with gr.Blocks(title="MLOps Monitoring Dashboard") as demo:
    gr.Markdown("# 📊 Dashboard Monitoraggio Sentiment Model (Twitter-RoBERTa)")
    gr.Markdown(
        "Osservabilità del modello: andamento metrica F1 su batch sequenziali, "
        "rilevamento drift e ripristino post-retraining."
    )

    status_box = gr.Textbox(label="Stato del Modello in Produzione", interactive=False)

    with gr.Row():
        refresh_btn = gr.Button("🔄 Aggiorna Metriche", variant="primary")

    gr.Markdown("### Andamento Temporale dell'Indice F1")
    plot = gr.LinePlot(
        value=load_metrics_data()[0],
        x="timestamp",
        y="f1_score",
        color="event_type",
        title="F1-Score per Batch e Tipologia di Evento",
        y_lim=[0.0, 1.0],
        tooltip=["batch_id", "model_id", "f1_score", "is_drifted"],
        height=380,  # <-- Rimosso 'width=850'
    )

    gr.Markdown("### Storico Run e Log Eventi")
    table = gr.DataFrame(value=load_metrics_data()[0], interactive=False)

    def update_view():
        data, status = load_metrics_data()
        return data, data, status

    refresh_btn.click(fn=update_view, outputs=[plot, table, status_box])
    demo.load(fn=update_view, outputs=[plot, table, status_box])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
