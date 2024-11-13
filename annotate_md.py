import datetime

import nltk
from langchain_core.language_models import BaseChatModel

from dotenv import load_dotenv
import os
import data
import experiments
import format
import json
from data.pet import PetDictExporter, PetJsonExporter
from experiments import sampling, parse
from format_efficiency import chat_model
from promptgen import model



def main():

    # Load sentence tokenizer if necessary
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    load_dotenv()
    #model_name = "gpt-4o-2024-08-06"
    model_name = "gpt-4o-mini"
    #model_name = "gpt-3.5-turbo"
    #model_name = "local_llm/llama3.1:70b"

    date_formatted = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if model_name.startswith("local_llm"):
        storage = f"res/answers/local_llm-llama3.1-70b/annotate/{date_formatted}.json"
    else:
        storage = f"res/answers/{model_name}/annotate-md/{date_formatted}.json"

    file_name = "1_5_3 Documentation and Justification Requirements"
    storage = f"res/data/annotate/pipe/2_md/{file_name}"
    chat_model: BaseChatModel = experiments.chat_model_for_name(model_name)
    file_path = file_name + ".txt"
    file_json = f"res/data/annotate/pipe/blank/{file_name}.jsonl"
    import_txt = data.AnnotateImporter(file_path)
    pet_json_exporter = PetJsonExporter(file_json)
    pet_json_exporter.export(import_txt.get_pedDoc())
    experiment = annotate(chat_model=chat_model, file_name=file_json, storage=storage)





    #experiments.print_experiment_results(storage, importer, verbose=True)
def annotate(chat_model: BaseChatModel, file_name: str, storage):

    importer = data.PetImporter(file_name)
    formatters = [
        format.IterativePetMentionListingFormattingStrategy(
            ["mentions"],
            "activity",
            context_tags=[],
            # prompt="pet/md/iterative/with_explanation/activity.txt",
        ),
        format.IterativePetMentionListingFormattingStrategy(
            ["mentions"],
            "actor",
            context_tags=["activity"],
            # prompt="pet/md/iterative/with_explanation/actor.txt",
        ),
        format.IterativePetMentionListingFormattingStrategy(
            ["mentions"],
            "activity data",
            context_tags=["activity", "actor"],
            # prompt="pet/md/iterative/with_explanation/activity_data.txt",
        ),
        format.IterativePetMentionListingFormattingStrategy(
            ["mentions"],
            "further specification",
            context_tags=["activity", "actor", "activity data"],
            # prompt="pet/md/iterative/with_explanation/further_specification.txt",
        ),
        format.IterativePetMentionListingFormattingStrategy(
            ["mentions"],
            "xor gateway",
            context_tags=[
                "activity",
                "actor",
                "activity data",
                "further specification",
            ],
            # prompt="pet/md/iterative/with_explanation/xor_gateway.txt",
        ),
        format.IterativePetMentionListingFormattingStrategy(
            ["mentions"],
            "condition specification",
            context_tags=[
                "activity",
                "actor",
                "activity data",
                "further specification",
                "xor gateway",
            ],
            # prompt="pet/md/iterative/with_explanation/condition_specification.txt",
        ),
        format.IterativePetMentionListingFormattingStrategy(
            ["mentions"],
            "and gateway",
            context_tags=[
                "activity",
                "actor",
                "activity data",
                "further specification",
                "xor gateway",
                "condition specification",
            ],
            # prompt="pet/md/iterative/with_explanation/and_gateway.txt",
        ),
    ]

    print(f"Using model: {chat_model.name}")
    num_shots = 0

    experiments.experiment(
        importer=importer,
        formatters=formatters,
        model_name=chat_model.name,
        chat_model=chat_model,
        storage=storage,
        num_shots=num_shots,
        dry_run=False,
    )
    return experiments

main()