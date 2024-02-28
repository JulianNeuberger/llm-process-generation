import datetime
import typing

import nltk

import data
import experiments
import format
from data import base


def select_documents(
        src: typing.List[base.DocumentBase], indexes: typing.List[int]
) -> typing.List[base.DocumentBase]:
    if len(indexes) == 0:
        return src

    result = []
    for idx in indexes:
        result.append(src[idx])
    return result


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

        num_shots = 1
        model_name = "gpt-4-0125-preview"

        formatter = format.PetTagFormattingStrategy()
        importer = data.VanDerAaImporter("res/data/van-der-aa/datacollection.csv")
        folds = [{"train": [], "test": [d.id for d in importer.do_import()[1:5]]}]

        # # formatter = format.QuishpiListingFormattingStrategy(["mentions"])
        # # importer = data.QuishpiImporter("res/data/quishpi", exclude_tags=["entity"])
        # # folds = [{"train": [], "test": ["7-1_calling_leads"]}]

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
