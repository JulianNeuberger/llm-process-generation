import datetime

import nltk
from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel

import data
import experiments
import format
from experiments import sampling

if __name__ == "__main__":

    def main():
        load_dotenv()

        # Load sentence tokenizer if necessary
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")

        num_shots = 1

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
        storage = f"res/answers/{model_name}/pet-re/{date_formatted}.json"
        # storage = f"res/answers/{model_name}/pet-re/2024-03-11_13-38-18.json"

        # formatter = format.PetMentionListingFormattingStrategy(["mentions"])
        importer = data.PetImporter("res/data/pet/all.new.jsonl")
        # folds = [
        #     {
        #         "train": [d.id for d in importer.do_import() if d.id != "doc-6.1"],
        #         "test": ["doc-6.1"],
        #     }
        # ]
        folds = sampling.generate_folds(
            importer.do_import(), num_shots, strategy="similarity"
        )

        # formatters = [format.PetRelationListingFormattingStrategy(steps=["relations"])]
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

        print("Using folds:")
        print("------------")
        for fold in folds:
            print(fold)
        print("------------")

        chat_model: BaseChatModel = experiments.chat_model_for_name(model_name)
        print(f"Using model: {chat_model.name}")

        experiments.experiment(
            importer=importer,
            formatters=formatters,
            model_name=model_name,
            chat_model=chat_model,
            storage=storage,
            num_shots=num_shots,
            dry_run=False,
            folds=folds,
        )

        experiments.print_experiment_results(
            storage, importer, verbose=True, print_only_tags=["same gateway"]
        )

    main()
