import typing

import format
from experiments.parse import parse_file, print_scores, print_scores_by_step, get_scores
import data
from collections import defaultdict
from data.pet import PetMention


def main():
    result_file1 = "res/data/annotate/pipe/run_1/5_annotate_re_extract/Quality_Assurance_in_Product_Development.json"
    # result_file2 = "res/answers/gpt-4o-mini/annotate/2024-10-17_16-24-29.json"
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
    importer = data.PetImporter("res/answers/gpt-4o-mini/annotate/results.jsonl")
    # pred_doc, steps = get_predicted_doc(result_file1, importer)
    # next_doc, steps = get_predicted_doc(result_file2, importer)
    # data.PetJsonExporter("res/answers/gpt-4o-2024-05-13/annotate-re/test.jsonl").export([next_doc])

    importing = data.PetImporter("res/data/annotate/tests/Inquiry_Offer_Order.jsonl")
    doc = importing.do_import()
    double_dict = get_double_assigned_tokens(doc[0])
    print("before")
    for x in doc[0].mentions:
        print(x)
    #for k, v in double_dict.items():
    #    print("index: ", k, " | type: ", v)
    remove_mention(doc[0], 4, "activity")
    print("after")
    for x in doc[0].mentions:
        print(x)
    remove_mention(doc[0], 2, "activity")
    print("after")
    for x in doc[0].mentions:
        print(x)

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


def get_double_assigned_token_indices(petDocument: data.PetDocument):
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


def get_double_assigned_tokens(petDocument: data.PetDocument) -> dict[int, typing.List["mentions"]]:
    pet_mention = petDocument.mentions
    indices_dict = defaultdict(list)
    for mentions in pet_mention:
        for i in mentions.token_document_indices:
            indices_dict[i].append(mentions.type)
    double_dict = defaultdict(list)
    for k, v in indices_dict.items():
        if len(v) >= 2:
            double_dict[k] = v
    return double_dict


def remove_mention(pet_document: data.PetDocument, mention_index: int, mention_type: str):
    pet_mentions = pet_document.mentions
    for mention in pet_mentions:
        if mention.type == mention_type:
            if mention_index in mention.token_document_indices:
                if len(mention.token_document_indices) == 1:
                    pet_mentions.remove(mention)
                    break
                elif len(mention.token_document_indices) == 2:
                    new_token_document_indices = tuple(x for x in mention.token_document_indices if x != mention_index)
                    pet_mentions.remove(mention)
                    pet_mentions.append(PetMention(mention_type, new_token_document_indices))
                    break
                else:
                    count_lower = []
                    count_higher = []
                    for i in mention.token_document_indices:
                        if i < mention_index:
                            count_lower.append(i)
                        elif i > mention_index:
                            count_higher.append(i)
                    new_token_document_indices = tuple(count_higher) if len(count_higher) > len(count_lower) else tuple(count_lower)
                    pet_mentions.remove(mention)
                    pet_mentions.append(PetMention(mention_type, new_token_document_indices))
                    break
    pet_document.mentions = pet_mentions


def double_assigned_remove(double_assigned: dict[int, typing.List["mentions"]], pet_document: data.PetDocument):
    mention_set = {
        "activity",
        "actor",
        "activity data",
        "xor gateway",
        "condition specification",
        "and gateway",
        "further specification"
    }
    for index, mention_types in double_assigned.items():
        for mention in mention_set:
            if mention in mention_types:
                mention_types.remove(mention)
                for x in mention_types:
                    remove_mention(pet_document, index, x)
                break

