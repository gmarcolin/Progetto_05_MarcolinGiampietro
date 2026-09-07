import gradio as gr
from huggingface_hub import InferenceClient
import yaml
import os

# Carichiamo la configurazione
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Usiamo il token dai segreti di sistema
HF_TOKEN = os.getenv("HF_TOKEN")
model_id = config['model']['hub_id']

# Client per le API Serverless di Hugging Face
client = InferenceClient(model=model_id, token=HF_TOKEN)


def predict(text):
    try:
        # Chiamata API al Model Hub
        response = client.text_classification(text)
        # Prendiamo il risultato con lo score più alto
        prediction = max(response, key=lambda x: x['score'])
        return f"Label: {prediction['label']} (Conf: {prediction['score']:.2f})"
    except Exception as e:
        err = f"Errore nell'interrogare il Model Hub: {str(e)}. "
        err += f"Assicurati che il modello sia pubblico o che il token sia corretto."
        return err


# Interfaccia Gradio
demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(placeholder="Inserisci un tweet qui..."),
    outputs="text",
    title="Sentiment Analysis - Serverless Mode",
    description=f"Questa app interroga il modello registrato su: {model_id}"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
