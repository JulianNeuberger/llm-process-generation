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
        storage = f"res/answers/pet/{date_formatted}.json"
        # storage = f"res/answers/pet/2024-02-27_13-29-40.json"

        num_shots = 0
        model_name = "gpt-4-0125-preview"

        # formatter = format.PetMentionListingFormattingStrategy(["mentions"])
        # formatter = format.PetTagFormattingStrategy()
        formatter = format.PetReferencesFormattingStrategy(["relations"])
        importer = data.PetImporter("res/data/pet/all.new.jsonl")
        folds = sampling.generate_folds(importer.do_import(), num_shots)
        print("Using folds:")
        print("------------")
        for fold in folds:
            print(fold)
        print("------------")

        experiments.experiment(
            importer=importer,
            formatter=formatter,
            model_name=model_name,
            storage=storage,
            num_shots=num_shots,
            dry_run=False,
            folds=folds,
        )

        experiments.print_experiment_results(storage, importer, verbose=True)

    main()
