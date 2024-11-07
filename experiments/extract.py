import typing

import eval
import format
import experiments
import data
from experiments.parse import parse_file, print_scores, print_scores_by_step, get_scores
import data


def main():
    #result_file1 = "res/answers/gpt-4o-mini/annotate/2024-10-18_08-58-23.json"
    #result_file2 = "res/answers/gpt-4o-mini/annotate/2024-10-17_16-24-29.json"
    result_file2 = "res/answers/gpt-4o-2024-05-13/annotate-re/2024-10-24_15-35-53.json"
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
    #importer = data.PetImporter("res/answers/gpt-4o-mini/annotate/results.jsonl")
    #pred_doc, steps = get_predicted_doc(result_file1, importer)
    #next_doc, steps = get_predicted_doc(result_file2, importer)
    #data.PetJsonExporter("res/answers/gpt-4o-2024-05-13/annotate-re/test.jsonl").export([next_doc])

    importing = data.PetImporter("res/answers/gpt-4o-2024-05-13/annotate-re/test.jsonl")
    doc = importing.do_import()
    ((get_double_assigned_TokenIndices(doc[0])))


    #stats, missing = experiments.consensus_2([pred_doc], [next_doc], False, None, steps)
    #scores = get_scores([stats],False)
    #print_scores_by_step(scores)
    #print(missing)

    #stats, missing = experiments.consensus_2([next_doc], [pred_doc], False, None, steps)
    #scores = get_scores([stats], False)
    #print_scores_by_step(scores)
    #print(missing)





def get_predicted_doc(
        result_file: str,
        importer: data.PetImporter):

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
    return predicted_doc, steps


def get_double_assigned_TokenIndices(petDocument: data.PetDocument):
    petMention = petDocument.mentions
    indicesList = [0] * len(petDocument.tokens)
    for mentions in petMention:
        for i in mentions.token_document_indices:
            indicesList[i] += 1
    double_List = []
    for i in range(0, len(indicesList)):
        if indicesList[i] >= 2:
            double_List.append(i)
    return double_List




main()