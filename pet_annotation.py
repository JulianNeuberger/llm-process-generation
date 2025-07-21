"""
This module can be used to incrementally annotate PET Dataset.

Author: Ivan Khrop
Date: 16.07.2025
"""
# import dependencies
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from experiments.sampling import generate_folds
from experiments import chat_model_for_name
from data import PetDocument, PetImporter
import format
from annotate_sap_sam.hint_provider import HintsProvider
from eval.metrics import mentions_f1_stats, Stats
from tqdm import tqdm

from annotate_sap_sam.utils import (
    annnotate_pet_by_formatters,
    save_PetDocuments,
)


# define paths to data foldes and files
models_folder = "pet"
base_path = Path(__file__).parent.joinpath("res", "data", models_folder)
pet_models_json_file = base_path.joinpath("pet_models.jsonl")
output_models_json_file = base_path.joinpath("end-to-end-predict.jsonl")

prompts_path = Path(__file__).parent.joinpath("res", "prompts", "pet")

# upload environmental variables in the memory
load_dotenv()

# select a model that will be used for annotation
{
    # model_name = "local_llm/llama3.1:70b"
    # model_name = "gpt-4-turbo-2024-04-09"
    # model_name = "gpt-4o-2024-05-13"
    # model_name = "claude-3-sonnet-20240229"
    # model_name = "claude-3-opus-20240229"
    # model_name = "meta-llama/Meta-Llama-3-70B-Instruct"
    # model_name = "deepinfra/airoboros-70b"
    # model_name = "gpt-4-0125-preview"
    # model_name = "Qwen/Qwen1.5-72B-Chat"
    # model_name = "gpt-3.5-turbo-0125"
    # model_name = "mistral-large-latest"
    # model_name = "gpt-4o-mini"
    # model_name = "llama-3.1-70b-versatile"
    # model_name = "gemma2-9b-it"
    # model_name = "mixtral-8x7b-32768"
}
# create a model
model_name = "gpt-4o"
gpt_chat_model: BaseChatModel = chat_model_for_name(model_name, 0)

# special requirements
use_graph_model = False
iterative_strategy = False # specify True to use iterative prompt
num_shots = 3 # amount of examples to take
seed = 42 # just a randomizer seed

# set up flag for hints
HintsProvider.use_hint = use_graph_model

# create formatters to avoid type errors
formatters: list[format.base.BaseFormattingStrategy] = list()

# define files with respect to intention to use hints and iterative strategy
if iterative_strategy:
    # ====================================================================
    # mention detection
    md_iterative_path = prompts_path.joinpath("md", "iterative", "with_explanation")
    activity_prompt_file = md_iterative_path.joinpath("activity.txt")
    actor_prompt_file = md_iterative_path.joinpath("actor.txt")
    activity_data_prompt_file = md_iterative_path.joinpath("activity_data.txt")
    further_specification_prompt_file = md_iterative_path.joinpath("further_specification.txt")
    xor_gateway_prompt_file = md_iterative_path.joinpath("xor_gateway.txt")
    condition_specification_prompt_file = md_iterative_path.joinpath("condition_specification.txt")
    and_gateway_prompt_file = md_iterative_path.joinpath("and_gateway.txt")

    # entitiy resolution
    er_unified_prompt_file = prompts_path.joinpath("er", "long.txt")

    # relation extraction
    re_iterative_path = prompts_path.joinpath("re", "iterative")
    same_gateway_prompt_file = re_iterative_path.joinpath("same_gateway.txt")
    flow_prompt_file = re_iterative_path.joinpath("flow.txt")
    remaining_prompt_file = re_iterative_path.joinpath("remaining.txt")
    # ====================================================================
else:
    # ====================================================================
    md_unified_prompt_file = prompts_path.joinpath("md", "unified.txt")
    er_unified_prompt_file = prompts_path.joinpath("er", "long.txt") # use long_sap_sam.txt will also identify the same activities
    re_unified_prompt_file = prompts_path.joinpath("re", "long.txt")
    # ====================================================================


# The stepwise annotation process in implemented below
# If some steps can be skipped, turn a boolean condition to the False
# Otherwise specify all flags as True.
def create_fold() -> list[dict[str, list[PetDocument]]]:
    """Create folds to annotate PET Dataset from scratch using PET Documents."""
    # import ground truth samples
    ground_truth_importer = PetImporter(str(pet_models_json_file))
    pet_documents = ground_truth_importer.do_import()
    pet_documents_dict = {doc.id: doc for doc in pet_documents}

    # import prediction samples
    prediction_importer = PetImporter(str(output_models_json_file))
    prediction_documents = prediction_importer.do_import()
    prediction_documents_dict = {doc.id: doc for doc in prediction_documents}
    

    # create folds for annotation considering only identifiers
    folds_ids = generate_folds(
        documents=pet_documents, 
        num_examples=num_shots, 
        seed=seed, 
        strategy="similarity"
    )

    # create folds with PetDocuments
    folds_docs: list[dict[str, list[PetDocument]]] = list()
    for fold in folds_ids:
        # create a new fold with PetDocuments
        documents_fold = dict()

        # combine ground truth and prediction documents into a fold
        documents_fold["train"] = [pet_documents_dict[doc_id] for doc_id in fold["train"]]
        documents_fold["test"] = [prediction_documents_dict[doc_id] for doc_id in fold["test"]]
            
        # save the fold
        folds_docs.append(documents_fold)
    
    return folds_docs


# ====================================================================
# Step 1. Create Non-Annotated PET Dataset.
# ====================================================================
if False:
    # read models for annotation
    importer = PetImporter(str(pet_models_json_file))
    pet_documents = importer.do_import()
    
    for doc in pet_documents:
        # doc.mentions = list()
        # doc.entities = list()
        doc.relations = list()

    # save all the processed PetDocuments in jsonl file
    assert save_PetDocuments(
        documents=pet_documents, 
        path=output_models_json_file,
    ), "Models were not loaded."
    print(f"Empty PetDocuments are saved to {str(output_models_json_file)}")


# ====================================================================
# Step 2. Mention Detection step for models.
# ====================================================================
if False:
    # create formatters depending on strategy
    if iterative_strategy:
        formatters = [
            format.IterativePetMentionListingFormattingStrategy(
                ["mentions"],
                "activity",
                context_tags=[],
                prompt=str(activity_prompt_file),
            ),
            format.IterativePetMentionListingFormattingStrategy(
                ["mentions"],
                "actor",
                context_tags=["activity"],
                prompt=str(actor_prompt_file),
            ),
            format.IterativePetMentionListingFormattingStrategy(
                ["mentions"],
                "activity data",
                context_tags=["activity", "actor"],
                prompt=str(activity_data_prompt_file),
            ),
            format.IterativePetMentionListingFormattingStrategy(
                ["mentions"],
                "further specification",
                context_tags=["activity", "actor", "activity data"],
                prompt=str(further_specification_prompt_file),
            ),
            format.IterativePetMentionListingFormattingStrategy(
                ["mentions"],
                "xor gateway",
                context_tags=[
                    "activity",
                    "actor",
                    "activity data",
                    "further specification",
                ],
                prompt=str(xor_gateway_prompt_file),
            ),
            format.IterativePetMentionListingFormattingStrategy(
                ["mentions"],
                "condition specification",
                context_tags=[
                    "activity",
                    "actor",
                    "activity data",
                    "further specification",
                    "xor gateway",
                ],
                prompt=str(condition_specification_prompt_file),
            ),
            format.IterativePetMentionListingFormattingStrategy(
                ["mentions"],
                "and gateway",
                context_tags=[
                    "activity",
                    "actor",
                    "activity data",
                    "further specification",
                    "xor gateway",
                    "condition specification",
                ],
                prompt=str(and_gateway_prompt_file),
            ),
        ]
    else:
        formatters = [
            format.PetMentionListingFormattingStrategy(
                steps=["mentions"], # just extract mentions
                prompt=str(md_unified_prompt_file) # unified prompt for mention detection
            )
        ]


    # annotate according formatters
    md_extracted = annnotate_pet_by_formatters(
        folds=create_fold(),
        formatters=formatters,
        chat_model=gpt_chat_model,
        model_name=model_name,
        dry_run=False,
    )

    # save all the processed PetDocuments in jsonl file
    assert save_PetDocuments(
        documents=md_extracted, 
        path=output_models_json_file,
    ), "Models were not annotated with mentions."
    print(f"PetDocuments with mentions are saved to {str(output_models_json_file)}")


# ====================================================================
# Step 3. Entity Resolution step for models.
# ====================================================================
if False:
    # only one prompt, no iterations
    formatters = [
        format.PetEntityListingFormattingStrategy(
            steps=["entities"], # just extract entities
            prompt=str(er_unified_prompt_file) # unified prompt for entity resolution
    )]
    
    # annotate according formatters
    er_extracted = annnotate_pet_by_formatters(
        folds=create_fold(),
        formatters=formatters,
        chat_model=gpt_chat_model,
        model_name=model_name,
        dry_run=False,
    )

    # save all the processed PetDocuments in jsonl file
    assert save_PetDocuments(
        documents=er_extracted, 
        path=output_models_json_file
    ), "Entity Resolution was not finished."
    print(f"PetDocuments with entities are saved to {str(output_models_json_file)}")


# ====================================================================
# Step 4. Relation Extraction step for models.
# ====================================================================
if False:
    # create formatters depending on strategy
    if iterative_strategy:
        formatters = [
            format.PetIterativeRelationListingFormattingStrategy(
                steps=["relations"],
                only_tags=["same gateway"],
                prompt=str(same_gateway_prompt_file),
            ),
            format.PetIterativeRelationListingFormattingStrategy(
                steps=["relations"],
                only_tags=["flow"],
                prompt=str(flow_prompt_file),
            ),
            format.PetIterativeRelationListingFormattingStrategy(
                steps=["relations"],
                only_tags=[
                    "uses",
                    "actor performer",
                    "actor recipient",
                    "further specification",
                ],
                prompt=str(remaining_prompt_file),
            ),
        ]
    else:
        formatters = [
            format.PetRelationListingFormattingStrategy(
                steps=["relations"], # just extract relations
                prompt=str(re_unified_prompt_file) # unified prompt for relation extraction
            )
        ]
    
    # annotate according formatters
    re_extracted = annnotate_pet_by_formatters(
        folds=create_fold(),
        formatters=formatters,
        chat_model=gpt_chat_model,
        model_name=model_name,
        dry_run=False,
    )


    # save all the processed PetDocuments in jsonl file
    assert save_PetDocuments(
        documents=re_extracted, 
        path=output_models_json_file
    ), "Relation Extraction was not finished."
    print(f"PetDocuments with relations are saved to {str(output_models_json_file)}")


# ====================================================================
# Step 5. Post-Processing step for models.
# ====================================================================
if False:
    # remove entities that do not have mentions
    documents = PetImporter(str(output_models_json_file)).do_import()

    # apply post-processing
    for doc in tqdm(documents, desc="Post-Processing"):
        doc.entities = [entity for entity in doc.entities if len(entity.mention_indices) > 0]

    # save all the processed PetDocuments in jsonl file
    assert save_PetDocuments(
        documents=documents, 
        path=output_models_json_file
    ), "Post-Processing was not finished."
    print(f"PetDocuments after post-processing are saved to {str(output_models_json_file)}")

# ====================================================================
# Step 6. Evaluation of the results for Mention Detection
# ====================================================================
if True:
    # create importers and import documents
    ground_truth_importer = PetImporter(str(pet_models_json_file))
    ground_truth_documents = ground_truth_importer.do_import()

    # create importers and import documents
    prediction_importer = PetImporter(str(output_models_json_file))
    prediction_documents = prediction_importer.do_import()

    # Mentions Score
    md_results = mentions_f1_stats(
        predicted_documents=prediction_documents,
        ground_truth_documents=ground_truth_documents,
        verbose=False,
        print_only_tags=["activity", "actor", "activity data", "further specification", "xor gateway", "condition specification", "and gateway"]
    )

    final_md_stats = Stats(
        num_pred=sum([result.num_pred for result in md_results.values()]),
        num_ok=sum([result.num_ok for result in md_results.values()]),
        num_gold=sum([result.num_gold for result in md_results.values()]),
    )

    print("==" * 20)
    print(md_results)
    print("Final MD Scores:")
    print("Precision:", round(final_md_stats.precision, 3))
    print("Recall:", round(final_md_stats.recall, 3))
    print("F1:", round(final_md_stats.f1, 3))
    print("==" * 20)


# ====================================================================
# Step 7. Evaluation of the results for Entity Resolution
# ====================================================================
if True:
    # create importers and import documents
    ground_truth_importer = PetImporter(str(pet_models_json_file))
    ground_truth_documents_dict = {
        doc.id: doc for doc in ground_truth_importer.do_import()
    }

    # create importers and import documents
    prediction_importer = PetImporter(str(output_models_json_file))
    prediction_documents_dict = {
        doc.id: doc for doc in prediction_importer.do_import()
    }

    # process all the documents
    er_stats = Stats(num_pred=0, num_gold=0, num_ok=0)

    for doc_id, ground_truth_doc in ground_truth_documents_dict.items():
        # create mapping for ground truth mentions converting a mention into a tuple
        mentions_mapping, reverse_mantion_mapping = dict(), dict()
        for idx, mention in enumerate(ground_truth_doc.mentions):
            # create a mention full tuple
            mentions_mapping[hash(mention)] = idx
            reverse_mantion_mapping[idx] = hash(mention)

        # now collect relation hashes
        ground_truth_entities: set[tuple] = set()
        for entity in ground_truth_doc.entities:
            # get all mentions inside the entity
            encoded_mentions = {
                hash(reverse_mantion_mapping[idx]) for idx in entity.mention_indices
            }
            # create a key as a tuple of unique sorted hashes
            key = tuple(sorted(list(encoded_mentions)))
            ground_truth_entities.add(key)
        
        # now encode all predicted relations
        predicted_entities: set[tuple] = set()
        for entity in prediction_documents_dict[doc_id].entities:
            # get all mentions inside the entity
            encoded_mentions = set()

            # go over all mentions and check problems
            for idx in entity.mention_indices:
                if idx < len(prediction_documents_dict[doc_id].mentions):
                    encoded_mentions.add(
                        hash(prediction_documents_dict[doc_id].mentions[idx]) 
                    )
                else:
                    # simply put something there, probability is super small
                    encoded_mentions.add(int(1e9 + 7))

            # create a key as a tuple of unique sorted hashes
            key = tuple(sorted(list(encoded_mentions)))
            predicted_entities.add(key)
        
        # how we can check the intersection of these two sets and find the results
        er_stats.num_gold += len(ground_truth_entities)
        er_stats.num_pred += len(predicted_entities)
        er_stats.num_ok += len(predicted_entities.intersection(ground_truth_entities))

    # now we can show the results
    print("==" * 20)
    print("Final RE Scores:")
    print("Precision:", round(er_stats.precision, 3))
    print("Recall:", round(er_stats.recall, 3))
    print("F1:", round(er_stats.f1, 3))
    print("==" * 20)


# ====================================================================
# Step 8. Evaluation of the results for Relation Extraction
# ====================================================================
if True:
    # create importers and import documents
    ground_truth_importer = PetImporter(str(pet_models_json_file))
    ground_truth_documents_dict = {
        doc.id: doc for doc in ground_truth_importer.do_import()
    }


    # create importers and import documents
    prediction_importer = PetImporter(str(output_models_json_file))
    prediction_documents_dict = {
        doc.id: doc for doc in prediction_importer.do_import()
    }

    # process all the documents
    re_stats = Stats(num_pred=0, num_gold=0, num_ok=0)

    for doc_id, ground_truth_doc in ground_truth_documents_dict.items():
        # get original relations
        original_relations = ground_truth_doc.relations
        original_relations_buffer = original_relations.copy()

        # predicted releations
        predicted_relations = prediction_documents_dict[doc_id].relations

        # now collect relation hashes
        matches: int  = 0
        predicted_tokens: set[int] = set()
        original_tokens: set[int] = set()
        for relation in predicted_relations:

            # get tokens for both head and tail
            predicted_tokens = set()
            predicted_tokens.update(prediction_documents_dict[doc_id].mentions[relation.head_mention_index].token_document_indices)
            predicted_tokens.update(prediction_documents_dict[doc_id].mentions[relation.tail_mention_index].token_document_indices)
            
            match_found: bool = False
            # look for relation in
            for idx, original_relation in enumerate(original_relations_buffer):
                original_tokens = set()

                if original_relation.type == relation.type:
                    # get tokens
                    original_tokens.update(ground_truth_doc.mentions[original_relation.head_mention_index].token_document_indices)
                    original_tokens.update(ground_truth_doc.mentions[original_relation.tail_mention_index].token_document_indices)

                # check match
                if len(original_tokens.intersection(predicted_tokens)) > 0:
                    match_found = True
                    break
                    
            # check match and remove relation from the buffer
            if match_found:
                matches += 1
                original_relations_buffer.remove(original_relation)
            

        # how we can check the intersection of these two sets and find the results
        re_stats.num_gold += len(original_relations)
        re_stats.num_pred += len(predicted_relations)
        re_stats.num_ok += matches

    # now we can show the results
    print("==" * 20)
    print("Final RE Scores:")
    print("Precision:", round(re_stats.precision, 3))
    print("Recall:", round(re_stats.recall, 3))
    print("F1:", round(re_stats.f1, 3))
    print("==" * 20)
    