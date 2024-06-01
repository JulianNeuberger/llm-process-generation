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

        num_shots = 3

        # model_name = "gpt-4-0125-preview"
        model_name = "gpt-4o-2024-05-13"
        # model_name = "claude-3-sonnet-20240229"
        # model_name = "claude-3-opus-20240229"

        date_formatted = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        storage = f"res/answers/{model_name}/pet-er/{date_formatted}.json"
        # storage = f"res/answers/{model_name}/pet-er/2024-03-01_10-11-43.json"

        # formatter = format.PetMentionListingFormattingStrategy(["mentions"])
        importer = data.PetImporter("res/data/pet/all.new.jsonl")
        # folds = [{"train": [], "test": ["doc-6.1"]}]
        folds = sampling.generate_folds(
            importer.do_import(), num_shots, seed=42, strategy="similarity"
        )

        formatters = [format.PetEntityListingFormattingStrategy(steps=["entities"])]

        print("Using folds:")
        print("------------")
        for fold in folds:
            print(fold)
        print("------------")

        chat_model: BaseChatModel = experiments.chat_model_for_name(model_name)

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
            storage, importer, verbose=True, print_only_tags=["Activity Data", "Actor"]
        )

    main()
