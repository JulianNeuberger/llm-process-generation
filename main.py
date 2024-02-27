import datetime

import nltk

import data
import experiments
import format

if __name__ == "__main__":

    def main():
        # Load sentence tokenizer if necessary
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")

        date_formatted = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        storage = f"res/answers/quishpi-re/{date_formatted}.json"
        # storage = "res/answers/pet/2024-02-20_13-04-48.json"

        num_shots = 0
        model_name = "gpt-4-0125-preview"

        # formatter = format.VanDerAaListingFormattingStrategy(steps=["constraints"])
        importer = data.VanDerAaImporter("res/data/quishpi/")
        docs = importer.do_import()
        print(len(docs))

        # folds = [{"train": [], "test": [d.id for d in importer.do_import()[1:2]]}]
        #
        # # formatter = format.QuishpiListingFormattingStrategy(["mentions"])
        # # importer = data.QuishpiImporter("res/data/quishpi", exclude_tags=["entity"])
        # # folds = [{"train": [], "test": ["7-1_calling_leads"]}]
        #
        # experiments.experiment(
        #     importer=importer,
        #     formatter=formatter,
        #     model_name=model_name,
        #     storage=storage,
        #     num_shots=num_shots,
        #     dry_run=False,
        #     folds=folds,
        # )

        # experiments.print_experiment_results(storage, importer, verbose=True)

    main()
