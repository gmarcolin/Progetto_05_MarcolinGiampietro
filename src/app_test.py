from huggingface_hub import HfApi
import os

api = HfApi(token=os.environ["HF_TOKEN"])

info = api.model_info(
    "gm84/sentiment-analysis-deploy",
    expand=["inference", "inferenceProviderMapping"]
)

print(info)
