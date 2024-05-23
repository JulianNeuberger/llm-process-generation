import datetime

import langchain_openai
import nltk
from langchain_core.language_models import BaseChatModel

import data
import experiments
import format
from experiments import sampling

if __name__ == "__main__":

    def main():
        # Load sentence tokenizer if necessary
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")

        date_formatted = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        storage = f"res/answers/pet-er/{date_formatted}.json"
        # storage = f"res/answers/pet-er/2024-03-01_10-11-43.json"

        num_shots = 0
        model_name = "gpt-4-0125-preview"

        # formatter = format.PetMentionListingFormattingStrategy(["mentions"])
        importer = data.PetImporter("res/data/pet/all.new.jsonl")
        # folds = [{"train": [], "test": ["doc-6.1"]}]
        folds = sampling.generate_folds(importer.do_import(), num_shots)

        formatters = [format.PetEntityListingFormattingStrategy(steps=["entities"])]

        print("Using folds:")
        print("------------")
        for fold in folds:
            print(fold)
        print("------------")

        chat_model: BaseChatModel = langchain_openai.ChatOpenAI(
            model_name=model_name, temperature=0
        )

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
