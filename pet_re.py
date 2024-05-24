import datetime

import langchain_openai
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

        num_shots = 0
        model_name = "gpt-4-0125-preview"

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
        folds = sampling.generate_folds(importer.do_import(), num_shots)

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

        chat_model = experiments.chat_model_for_name(model_name)

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
