import gradio as gr
from huggingface_hub import InferenceClient
import yaml
import os

# Carichiamo la configurazione
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

model_id = config["model"]["hub_id"]

# Token HF configurato nei Secrets dello Space
#HF_TOKEN = os.getenv("HF_TOKEN")

# Diagnostica
#print(f"Model ID: {model_id}")
#print(f"HF_TOKEN presente: {HF_TOKEN is not None}")
#print(f"HF_TOKEN vuoto: {not HF_TOKEN if HF_TOKEN is not None else True}")

# Client HF Inference
client = InferenceClient(
    provider="hf-inference",
    api_key=os.environ["HF_TOKEN"]
)


def predict(text):
    try:
        response = client.text_classification(
            text,
            model=model_id
        )

        print("Risposta HF:", response)

        prediction = max(response, key=lambda x: x["score"])

        return (
            f"Label: {prediction['label']} "
            f"(Conf: {prediction['score']:.2f})"
        )

    except Exception as e:
        print(f"ERRORE: {type(e).__name__}: {repr(e)}")

        return (
            f"ERRORE: {type(e).__name__}: {repr(e)}"
        )


demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(placeholder="Inserisci un tweet qui..."),
    outputs="text",
    title="Sentiment Analysis - Serverless Mode",
    description=f"Questa app interroga il modello registrato su: {model_id}"
)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860
    )
