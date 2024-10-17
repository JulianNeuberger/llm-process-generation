import typing
import format

import data
from experiments.parse import parse_file, parse_experiment


def main():
    result_file = "res/answers/gpt-4o-mini/annotate/2024-10-03_19-08-36.json"
    importers = {
        "pet": data.PetImporter("res/data/annotate/Inquiry_Offer_Order.jsonl"),
        "quishpi-re": data.VanDerAaSentenceImporter("res/data/quishpi/csv"),
        "quishpi-md": data.QuishpiImporter("res/data/quishpi", exclude_tags=["entity"]),
        "van-der-aa-re": data.VanDerAaSentenceImporter(
            "res/data/van-der-aa/datacollection.csv"
        ),
        "van-der-aa-md": data.VanDerAaImporter(
            "res/data/van-der-aa/datacollection.csv"
        ),
        "analysis": data.PetImporter("res/data/pet/all.new.jsonl"),
    }
    importer = data.PetImporter("res/data/annotate/Inquiry_Offer_Order.jsonl")

    documents = importer.do_import()
    documents_by_id = {d.id: d for d in documents}

    experiment_results = parse_file(result_file)
    for experiment in experiment_results:
        for result in experiment.results:
            predicted_doc: typing.Optional[data.PetDocument] = None
            input_doc = documents_by_id[result.original_id]
            for formatter_class_name, steps, answer, prompt, args in zip(
                    result.formatters,
                    result.steps,
                    result.answers,
                    result.prompts,
                    result.formatter_args,
            ):

                formatter_class: typing.Type[format.BaseFormattingStrategy] = getattr(
                    format, formatter_class_name
                )
                formatter = formatter_class(steps, **args)
                partial_prediction = formatter.parse(input_doc, answer)
                if predicted_doc is None:
                    predicted_doc = partial_prediction.document
                else:
                    predicted_doc = predicted_doc + partial_prediction.document
            assert predicted_doc is not None
            data.PetJsonExporter("res/answers/gpt-4o-mini/annotate/result.jsonl").export([predicted_doc])
            print(predicted_doc)

main()