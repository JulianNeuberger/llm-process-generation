import datetime

import nltk

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
        # storage = f"res/answers/pet/{date_formatted}.json"
        storage = f"res/answers/pet/2024-02-29_16-41-29.json"

        num_shots = 1
        model_name = "gpt-4-0125-preview"

        # formatter = format.PetMentionListingFormattingStrategy(["mentions"])
        importer = data.PetImporter("res/data/pet/all.new.jsonl")
        train_docs = [d.id for d in importer.do_import() if d.id != "doc-6.1"]
        # folds = [{"train": train_docs, "test": ["doc-6.1"]}]
        folds = sampling.generate_folds(importer.do_import(), num_shots)

        formatters = [
            format.PetActivityListingFormattingStrategy(["mentions"]),
            format.PetActorListingFormattingStrategy(["mentions"]),
            format.PetDataListingFormattingStrategy(["mentions"]),
            format.PetFurtherListingFormattingStrategy(["mentions"]),
            format.PetXorListingFormattingStrategy(["mentions"]),
            format.PetConditionListingFormattingStrategy(["mentions"]),
            format.PetAndListingFormattingStrategy(["mentions"]),
        ]

        print("Using folds:")
        print("------------")
        for fold in folds:
            print(fold)
        print("------------")

        experiments.experiment(
            importer=importer,
            formatters=formatters,
            model_name=model_name,
            storage=storage,
            num_shots=num_shots,
            dry_run=False,
            folds=folds,
        )

        experiments.print_experiment_results(
            storage, importer, verbose=True, print_only_tags=["Activity Data"]
        )

    main()
