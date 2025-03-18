"""
This module can be used to incrementally annotate any text annotation in PET dataset format.
Adapt it to the structure of your document.

Author: Ivan Khrop
Date: 16.12.2024
"""
# import dependencies
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
import pandas as pd
from tqdm import tqdm
import experiments
import format

from annotate_sap_sam.utils import (
    text_to_PetDocumentView, # converts texts to PetDocument format 
    save_PetDocuments, # save PetDocuments in a file .jsonl
    annotate_by_formatters, # apply formatters for specific annotation
    MixedDataImporter, # handles both PET-Dataset and SAP-SAM-Dataset
    get_double_assigned_tokens, # tokens that were assigned to more than one mention
    double_assigned_remove, # remove mentions that are assigned to one token
)
from annotate_sap_sam.hint_provider import HintsProvider # provides hints for iterative annotation

# define paths to data foldes and files
models_folder = "sap sam"
base_path = Path(__file__).parent.joinpath("res", "data", "annotate", models_folder)
sap_sam_models_path = base_path.joinpath("sap_sam_models.xlsx")
sap_sam_json_file = base_path.joinpath("sap_sam_models.jsonl")
pet_models_json_file = base_path.joinpath("pet_models.jsonl")

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
model_name = "gpt-4o" #
# create a model
gpt_chat_model: BaseChatModel = experiments.chat_model_for_name(model_name, 0)

# special requirements
iterative_strategy = False # specify True to use iterative prompt
use_graph_model = True # specify True to use prompt with graph ground truth model
num_shots = 4 # amount of examples to take
seed = 42 # just a randomizer seed

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
    md_unified_prompt_file = prompts_path.joinpath("md", "unified_sap_sam.txt")
    er_unified_prompt_file = prompts_path.joinpath("er", "long.txt") # use long_sap_sam.txt will also identify the same activities
    re_unified_prompt_file = prompts_path.joinpath("re", "long_sap_sam.txt")
    # ====================================================================
# set up flag for hints
HintsProvider.use_hint = use_graph_model

# The stepwise annotation process in implemented below
# If some steps can be skipped, turn a boolean condition to the False
# Otherwise specify all flags as True.

# ====================================================================
# Step 1. Convert plain text decscription into a PetDocument.
# ====================================================================
if False:
    # read models for annotation
    # !!! TODO !!!
    # you can adjust this section for your input format
    # all models must be organized as dictionary {(model_name, model_id): text}
    models_df = pd.read_excel(io=sap_sam_models_path)
    text_models: dict[tuple[str, str], str] = dict()
    # convert each int text and id
    for raw in models_df.itertuples(index=False):
        model_name, models_id, _, _, _, _, model_text, _ = tuple(raw)
        text_models[(model_name, models_id)] = model_text

    # convert each model in PetDocument
    converted_to_pet_models = list()
    for model_name, model_id in tqdm(
        iterable=text_models.keys(), 
        desc="Converting models into PetDocument format"
    ):
        model_text = text_models[(model_name, model_id)]
        pet_document = text_to_PetDocumentView(text=model_text, 
                                               text_id=model_id,
                                               text_name=model_name
        )
        converted_to_pet_models.append(pet_document)
    
    # save all the obtained PetDocuments in jsonl file
    assert save_PetDocuments(
        documents=converted_to_pet_models, 
        path=sap_sam_json_file
    ), "Models were not converted into PetDocument format."
    print(f"PetDocumemts with tokens are saved to {str(sap_sam_json_file)}")


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
                prompt=activity_prompt_file,
            ),
            format.IterativePetMentionListingFormattingStrategy(
                ["mentions"],
                "actor",
                context_tags=["activity"],
                prompt=actor_prompt_file,
            ),
            format.IterativePetMentionListingFormattingStrategy(
                ["mentions"],
                "activity data",
                context_tags=["activity", "actor"],
                prompt=activity_data_prompt_file,
            ),
            format.IterativePetMentionListingFormattingStrategy(
                ["mentions"],
                "further specification",
                context_tags=["activity", "actor", "activity data"],
                prompt=further_specification_prompt_file,
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
                prompt=xor_gateway_prompt_file,
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
                prompt=condition_specification_prompt_file,
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
                prompt=and_gateway_prompt_file,
            ),
        ]
    else:
        formatters = [
            format.PetMentionListingFormattingStrategy(
                steps=["mentions"], # just extract mentions
                prompt=md_unified_prompt_file # unified prompt for mention detection
            )
        ]
    
    # create importer
    md_importer = MixedDataImporter(
        pet_path=pet_models_json_file, 
        sap_sam_path=sap_sam_json_file
    )

    # annotate according formatters
    md_extracted = annotate_by_formatters(
        mixed_data_importer=md_importer,
        formatters=formatters,
        chat_model=gpt_chat_model,
        model_name=model_name,
        dry_run=False,
        num_shots=num_shots,
        seed=seed,
    )

    # Check for correctness
    md_extracted_new = list()
    for doc in md_extracted:
        if len(doc.mentions) == 0:
            print(f"Mentions for document {doc.id} were not detected.")
        else:
            md_extracted_new.append(doc)

    # save all the processed PetDocuments in jsonl file
    assert save_PetDocuments(
        documents=md_extracted_new, 
        path=sap_sam_json_file
    ), "Models were not annotated with mentions."
    print(f"PetDocumemts with mentions are saved to {str(sap_sam_json_file)}")


# ====================================================================
# Step 3. Remove intersections in mentions.
# ====================================================================
if False:
    # read the models of SAP-SAM-Dataset
    importer = MixedDataImporter(
        pet_path=pet_models_json_file, 
        sap_sam_path=sap_sam_json_file
    )
    documents = importer._sap_sam_documents

    # check if there are any intersections (several mentions on the same token index)
    for document in tqdm(documents, desc="Removal of errors in mentions"):
        # check the document
        problematic_tokens = get_double_assigned_tokens(pet_document=document)
        # report the amount of problematic tokens
        print("Amount of problematic tokens:", len(problematic_tokens))
        # solve disambiguity in mentions
        double_assigned_remove(pet_document=document, double_assigned=problematic_tokens)
    
    # save all the obtained PetDocuments in jsonl file
    assert save_PetDocuments(
        documents=documents, 
        path=sap_sam_json_file
    ), "Models after correction of mentions were not saved."
    print(f"PetDocumemts after correction of mentions are saved to {str(sap_sam_json_file)}")


# ====================================================================
# Step 4. Entity Resolution step for models.
# ====================================================================
if False:
    # only one prompt, no iterations
    formatters = [
        format.PetEntityListingFormattingStrategy(
            steps=["entities"], # just extract entities
            prompt=er_unified_prompt_file # unified prompt for entity resolution
    )]
    
    # create importer
    er_importer = MixedDataImporter(
        pet_path=pet_models_json_file, 
        sap_sam_path=sap_sam_json_file
    )

    # annotate according formatters
    er_extracted = annotate_by_formatters(
        mixed_data_importer=er_importer,
        formatters=formatters,
        chat_model=gpt_chat_model,
        model_name=model_name,
        dry_run=False,
        num_shots=num_shots,
        seed=seed,
    )

    # Check for correctness
    er_extracted_new = list()
    for doc in er_extracted:
        if len(doc.entities) == 0:
            print(f"Entities for document {doc.id} were not detected.")
        else:
            er_extracted_new.append(doc)

    # save all the processed PetDocuments in jsonl file
    assert save_PetDocuments(
        documents=er_extracted_new, 
        path=sap_sam_json_file
    ), "Entity Resolution was not finished."
    print(f"PetDocumemts with entities are saved to {str(sap_sam_json_file)}")


# ====================================================================
# Step 5. Relation Extraction step for models.
# ====================================================================
if False:
    # create formatters depending on strategy
    if iterative_strategy:
        formatters = [
            format.PetIterativeRelationListingFormattingStrategy(
                steps=["relations"],
                only_tags=["same gateway"],
                prompt=same_gateway_prompt_file,
            ),
            format.PetIterativeRelationListingFormattingStrategy(
                steps=["relations"],
                only_tags=["flow"],
                prompt=flow_prompt_file,
            ),
            format.PetIterativeRelationListingFormattingStrategy(
                steps=["relations"],
                only_tags=[
                    "uses",
                    "actor performer",
                    "actor recipient",
                    "further specification",
                ],
                prompt=remaining_prompt_file,
            ),
        ]
    else:
        formatters = [
            format.PetRelationListingFormattingStrategy(
                steps=["relations"], # just extract relations
                prompt=re_unified_prompt_file # unified prompt for relation extraction
            )
        ]
    
    # create importer
    re_importer = MixedDataImporter(
        pet_path=pet_models_json_file, 
        sap_sam_path=sap_sam_json_file
    )

    # annotate according formatters
    re_extracted = annotate_by_formatters(
        mixed_data_importer=re_importer,
        formatters=formatters,
        chat_model=gpt_chat_model,
        model_name=model_name,
        dry_run=False,
        num_shots=num_shots,
        seed=seed,
    )

    # Check for correctness
    re_extracted_new = list()
    for doc in re_extracted:
        if len(doc.relations) == 0:
            print(f"Relations for document {doc.id} were not detected.")
        else:
            re_extracted_new.append(doc)

    # save all the processed PetDocuments in jsonl file
    assert save_PetDocuments(
        documents=re_extracted_new, 
        path=sap_sam_json_file
    ), "Relation Extraction was not finished."
    print(f"PetDocumemts with relations are saved to {str(sap_sam_json_file)}")


# ====================================================================
# Step 6. Post-Processing step for models.
# ====================================================================
if False:
    # create importer
    importer = MixedDataImporter(
        pet_path=pet_models_json_file, 
        sap_sam_path=sap_sam_json_file
    )

    # apply post-processing
    for doc in tqdm(importer._sap_sam_documents, desc="Post-Processing"):
        doc.entities = [entity for entity in doc.entities if len(entity.mention_indices) > 0]

    # save all the processed PetDocuments in jsonl file
    assert save_PetDocuments(
        documents=importer._sap_sam_documents, 
        path=sap_sam_json_file
    ), "Post-Processing was not finished."
    print(f"PetDocumemts after post-processing are saved to {str(sap_sam_json_file)}")