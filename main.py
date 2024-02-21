import datetime

import nltk

import data
import experiments
import format

if __name__ == "__main__":

    def main():
        # Load sentence tokenizer if necessary
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

        date_formatted = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        storage = f"res/answers/vanderaa/{date_formatted}.json"
        # storage = "res/answers/pet/2024-02-20_13-04-48.json"

        num_shots = 0
        # formatter = format.ReferencesFormattingStrategy(["relations"])
        formatter = format.VanDerAaListingFormattingStrategy(steps=["constraints"])
        model_name = "gpt-4-0125-preview"

        # importer = data.PetImporter("res/data/pet/all.new.jsonl")
        importer = data.VanDerAaImporter("res/data/van-der-aa/datacollection.csv")
        # with open("res/data/pet/folds.json", "r", encoding="utf8") as f:
        #   folds = json.load(f)

        folds = [{"train": [], "test": [d.id for d in importer.do_import()]}]

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
