import pandas as pd

from matplotlib import pyplot as plt
from transformers import BertTokenizer
from pathlib import Path

from annotate_sap_sam.utils import  MixedDataImporter # handles both PET-Dataset and SAP-SAM-Dataset
from data.pet import PetDocument

# Load the pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# paths and folders
models_folder = "sap sam"
base_path = Path(__file__).parent.joinpath("res", "data")
sap_sam_json_file = base_path.joinpath("annotate", models_folder, "sap_sam_models.jsonl")
pet_models_json_file = base_path.joinpath("pet", "all_new.jsonl")

# create importer
importer = MixedDataImporter(
    pet_path=pet_models_json_file, 
    sap_sam_path=sap_sam_json_file
)

# ==================================
# Tokens
# ==================================
def token_stats(documents: list[PetDocument], dataset_name: str, n_max: int = 512):
    """Print histogram with of tokens for set of docuemnts."""
    data: list[int] = list()

    # go over all rows
    for document in documents:
        _id = document.id
        text = document.text

        # Tokenize the text
        tokens = tokenizer.tokenize(text)

        # Check the number of tokens
        token_count = len(tokens)

        # Update the maximum number of tokens
        n_max = max(n_max, token_count)
        data.append(token_count)

        if token_count > 512:
            print(f"ID: {_id}, Tokens: {token_count}")

    # Display the tokens and their count
    print("Maximal amount of tokens:", n_max)

    plt.hist(data, bins=30, alpha=0.5, color="blue", edgecolor="black")
    plt.xlabel("Number of tokens")
    plt.ylabel("Frequency")
    plt.title(f"Amount of tokens in {dataset_name}")
    plt.show()

# Statistics about tokens
if False:
    token_stats(importer._pet_documents, dataset_name="PET Dataset")
    token_stats(importer._sap_sam_documents, dataset_name="SAP SAM Dataset")


# ==================================
# Actors 
# ==================================
def get_words_for_actors(documents: list[PetDocument]) -> set[str]:
    actors_words: list[str] = list()

    # collect all substring
    for document in documents:
        for mention in document.mentions:
            if mention.type == "actor":
                text = mention.text(document=document).lower()
                words = [word.strip() for word in text.split() if word not in {"a", "an", "the"}]
                actors_words.extend(words)
    
    return set(actors_words)

# Check Actors
def actors_words(documents: list[PetDocument], dataset_name: str):
    """Create word cloud with actors for a dataset."""
    actors_names: list[str] = list()

    # collect all substring
    for document in documents:
        for mention in document.mentions:
            if mention.type == "actor":
                text = mention.text(document=document).lower()
                words = [word.strip() for word in text.split() if word not in {"a", "an", "the"}]
                actors_names.append(" ".join(words))

    actors_counter: dict[str, int] = dict()

    for actor in actors_names:
        if actor not in actors_counter:
            actors_counter[actor] = 0

        actors_counter[actor] += 1
    
    df = pd.DataFrame.from_dict(data=actors_counter, orient="index", columns=["Count"]).reset_index(drop=False)
    df["Percent"] = 100.00 * df["Count"] / df["Count"].sum()
    df = df.round(2)

    # print results
    print(f"Actors of {dataset_name}")
    print(df.describe())
    print()
    for row in df.itertuples(index=False):
        text, count, percent = tuple(row)
        # if more than 5 mentions, then show it
        if percent > 1.00:
            print(f"Word: {text}, Count: {count}, Percent: {percent}")
    print()

# Print the most frequent actors
if False:
    actors_words(importer._pet_documents, dataset_name="PET Dataset")
    actors_words(importer._sap_sam_documents, dataset_name="SAP SAM Dataset")

# Check intersections between actors
if True:
    pet_actors = get_words_for_actors(importer._pet_documents)
    sap_sam_actors = get_words_for_actors(importer._sap_sam_documents)

    #  check size of intersection
    intersection = pet_actors.intersection(sap_sam_actors)
    print("Intersection regarding PET size:", round(len(intersection) / len(pet_actors), 3))
    print("Intersection regarding SAP SAM size:", round(len(intersection) / len(sap_sam_actors), 3))


# ==================================
# Relations
# ==================================
def count_relations(documents: list[PetDocument], dataset_name: str):
    """Count all relations by types in the dataset."""
    relations: dict[str, int] = dict()

    # count all relations
    for document in documents:
        for relation in document.relations:
            if relation.type not in relations:
                relations[relation.type] = 0
            
            relations[relation.type] += 1
    
    # create a DataFrame
    df = pd.DataFrame.from_dict(data=relations, orient="index", columns=["Count"])
    df["Percent"] = 100.00 * df["Count"] / df["Count"].sum()
    df = df.round(2)

    # print results
    print(f"Relations of {dataset_name}")
    print(df)
    print()


if False:
    count_relations(importer._pet_documents, dataset_name="PET Dataset")
    count_relations(importer._sap_sam_documents, dataset_name="SAP SAM Dataset")


# ==================================
# Entities
# ==================================
def count_entities(documents: list[PetDocument], dataset_name: str):
    entities: dict[str, int] = dict()

    for document in documents:
        for mention in document.mentions:

            if mention.type not in entities:
                entities[mention.type] = 0
            
            entities[mention.type] += 1
            
        """
        for entity in document.entities:
            # get tag
            tag = entity.get_tag(document=document)
            # if tag is actor or activity
            if tag in {"actor", "activity data"}:
                if tag not in entities:
                    entities[tag] = 0
                entities[tag] += 1
        """
    
     # create a DataFrame
    df = pd.DataFrame.from_dict(data=entities, orient="index", columns=["Count"])
    df["Percent"] = 100.00 * df["Count"] / df["Count"].sum()
    df = df.round(2)

    # print results
    print(f"Entities of {dataset_name}")
    print(df)
    print()

if False:
    count_entities(importer._pet_documents, dataset_name="PET Dataset")
    count_entities(importer._sap_sam_documents, dataset_name="SAP SAM Dataset")