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
        storage = f"res/answers/quishpi-md/{date_formatted}.json"
        # storage = f"res/answers/quishpi-md/2024-03-01_08-37-31.json"

        num_shots = 3
        model_name = "gpt-4-0125-preview"

        importer = data.QuishpiImporter("res/data/quishpi", exclude_tags=["entity"])
        # folds = [{"train": [], "test": ["20818304_rev1"]}]
        folds = sampling.generate_folds(importer.do_import(), num_shots)

        formatters = [format.QuishpiMentionListingFormattingStrategy(["mentions"])]

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

        experiments.print_experiment_results(storage, importer, verbose=True)

    main()
