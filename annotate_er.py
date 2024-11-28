import datetime

import langchain_openai
import nltk
from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel

import data
import experiments
import format
from experiments import sampling


def main():
    load_dotenv()

    # Load sentence tokenizer if necessary
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    #storage = "res/data/annotate/pipe/run_1/4_annotate_er/Contract Review and Approval Process.json"
    file_name = "res/data/annotate/pipe/run_1/3_annotate_md_extract/Contract Review and Approval Process.json"
    model_name = "gpt-4o"
    chat_model: BaseChatModel = experiments.chat_model_for_name(model_name, 1)
    #annotate_er(chat_model, file_name, storage)

def annotate_er(chat_model: BaseChatModel, file_name: str, storage: str):
    importer = data.PetImporter(file_name)
    formatters = [format.PetEntityListingFormattingStrategy(steps=["entities"])]
    num_shots = 0
    print(f"Using model : {chat_model.name}")
    experiments.experiment(
        importer=importer,
        formatters=formatters,
        model_name=chat_model.name,
        chat_model=chat_model,
        storage=storage,
        num_shots=num_shots,
        dry_run=False,
        folds=None
    )
    return experiments, importer

main()

