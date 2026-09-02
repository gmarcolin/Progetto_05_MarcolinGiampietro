import gradio as gr
from transformers import pipeline
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

try:
    classifier = pipeline("sentiment-analysis", model=config['model']['hub_id'])
except:
    classifier = pipeline("sentiment-analysis", model=config['model']['base_name'])

def predict(text):
    res = classifier(text)[0]
    return f"{res['label']} ({res['score']:.2f})"

gr.Interface(fn=predict, inputs="text", outputs="text").launch()