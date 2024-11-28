import datetime

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

    num_shots = 0

    # model_name = "gpt-4-turbo-2024-04-09"
    model_name = "gpt-4o-2024-05-13"
    # model_name = "claude-3-sonnet-20240229"
    # model_name = "claude-3-opus-20240229"
    # model_name = "meta-llama/Meta-Llama-3-70B-Instruct"
    # model_name = "deepinfra/airoboros-70b"
    # model_name = "gpt-4-0125-preview"
    # model_name = "Qwen/Qwen1.5-72B-Chat"
    # model_name = "gpt-3.5-turbo-0125"

    date_formatted = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    storage = f"res/answers/{model_name}/annotate-re/{date_formatted}.json"
    file_name = "1_5_3 Documentation and Justification Requirements"
    #file_json = f"res/data/annotate/{file_name}.jsonl"
    file_json = "res/answers/gpt-4o-mini/annotate/results.jsonl"
    # formatter = format.PetMentionListingFormattingStrategy(["mentions"])
    chat_model: BaseChatModel = experiments.chat_model_for_name(model_name)
    annotate_re(chat_model, file_json, storage)


def annotate_re(chat_model: BaseChatModel, file_name: str, storage):
    importer = data.PetImporter(file_name)
    formatters = [
        format.PetIterativeRelationListingFormattingStrategy(
            ["relations"],
            "pet/re/iterative/same_gateway.txt",
            only_tags=["same gateway"],
        ),
        format.PetIterativeRelationListingFormattingStrategy(
            ["relations"],
            "pet/re/iterative/flow.txt",
            only_tags=["flow"],
        ),
        format.PetIterativeRelationListingFormattingStrategy(
            ["relations"],
            "pet/re/iterative/remaining.txt",
            only_tags=[
                "uses",
                "actor performer",
                "actor recipient",
                "further specification",
            ],
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
    return experiments, importer
    #experiments.print_experiment_results(storage, importer, verbose=True, print_only_tags=["same gateway"])


