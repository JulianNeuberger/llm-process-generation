"""
This module contains fuctions that can be used for organizing a pipeline for annotation of SAP SAM Models.

Author: Ivan Khrop
Date: 12.12.2024
"""
# import basic packages
import nltk
import json
from pathlib import Path
from typing import Union
import random
from tqdm import tqdm
from langchain_core.language_models import BaseChatModel

# import customized packages
from data.pet import PetToken, PetDocument, PetMention
from data import PetImporter
from data.base import BaseImporter
from experiments.sampling import random_sample_examples
from format import BaseFormattingStrategy
from experiments.iterative import run_iterative_document_prompt_getting_doc # depends on the amount of formatters, can be used non-iterative 

# check if the sentence tokenizer models are downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    # if not, then download
    print("Punkt tokenizer not found. Downloading now...")
    nltk.download('punkt')


def save_PetDocuments(documents: list[PetDocument], path: Union[str, Path]) -> bool:
    """
    Write documents into a file using json-line format.

    Parameters
    ----------
    documents: list[PetDocument]
        Documents that must be saved.
    path: Union[str, Path]
        Path to the file that will store the json-lines.

    Returns
    --------
    bool
        Flag. True when the operation succeeded, otherwise False.
    """
    try:
        with open(path, "w") as file:
            for document in documents:
                # create a dictionary from document
                document_dict = document.to_dict()
                # save document as a json-line in the file
                file.write(json.dumps(document_dict) + "\n")
    except IsADirectoryError:
        print("Expected a file, but found a directory.")
        return False
    except OSError as e:
        print(f"An OS error occurred: {e}")
        return False

    # ok, return true
    return True


def text_to_PetTokens(text: str) -> list[PetToken]:
    """
    Convert a plain text into a list of Pet Tokens.

    Parameters
    ----------
    text: str
        Text to convert.
    
    Returns
    -------
    list[PetToken]
        List of Pet Tokens for a text.
    """
    # create variables
    word_counter = 0
    sentence_counter = 0
    tokens: list[PetToken] = list()

    # extract sentences
    sentences = nltk.tokenize.sent_tokenize(text=text, language="english")

    # for each sentence do: POS-Tagging
    for sentence in sentences:
        # extract words and assign POS-Tags
        words = nltk.tokenize.word_tokenize(text=sentence)
        pos_tags = nltk.tag.pos_tag(words)

        # for each word, create a PetToken
        for word, tag in pos_tags:
            # create a PetToken
            tokens.append(
                PetToken(text=word,
                         index_in_document=word_counter, 
                         pos_tag=tag, 
                         sentence_index=sentence_counter,
                )
            )
            # increment a word counter
            word_counter += 1
        # increment a sentence counter
        sentence_counter += 1

    # return results
    return tokens


def text_to_PetDocumentView(
        text: str, 
        text_id: str,
        text_name: str, 
        category: str = ""
) -> PetDocument:
    """
    Create a simple PetDocument for text without mentions, entities and relations.

    Parameters
    ----------
    text: str
        Text to convert into PetDocument format.
    text_id: str
        Unique identifier of the text.
    text_id: str
        Unique name of the text.
    category: str = ""
        Text category if it exists.

    Returns
    -------
    PetDocument
        PetDocument instance without mentions, entities and relations.
    """
    # get tokens
    tokens = text_to_PetTokens(text=text)
    # create document and return it
    return PetDocument(
        text=text,
        name=text_name,
        id=text_id,
        category=category,
        tokens=tokens,
        mentions=list(),
        entities=list(),
        relations=list(),
    )


# class for processing both PET and SAP SAM Datasets simultaneously
class MixedDataImporter(BaseImporter[PetDocument]):
    """
    Class combines two importers: PET-Dataset Importer and SAP-SAM-Dataset Importer.
    It's necessary to maintain them separately through the annotation procedure.

    Attributes
    ----------
    _pet_importer: PetImporter
        Importer responsible for PET-Dataset.
    _sap_sam_importer: PetImporter
        Importer responsible for SAP-SAM-Dataset.
    """
    # define fields
    _pet_importer: PetImporter
    _sap_sam_importer: PetImporter
    _pet_documents: list[PetDocument]
    _sap_sam_documents: list[PetDocument]

    def __init__(self, pet_path: Union[str, Path], sap_sam_path: Union[str, Path]):
        """
        Initialize importer providing paths to both datasets: PET and SAP SAM.

        Parameters
        ----------
        pet_path: Union[str, Path]
            Path to PET-Dataset.
        sap_sam_path: Union[str, Path]
            Path to SAP-SAM-Dataset.
        """
        self._pet_importer = PetImporter(pet_path)
        self._sap_sam_importer = PetImporter(sap_sam_path)
        # read PET-Dataset
        self._pet_documents = self._pet_importer.do_import()
        # read SAP-SAM Dataset
        self._sap_sam_documents = self._sap_sam_importer.do_import()
    
    def do_import(self) -> list[PetDocument]:
        # just return available documents
        return self._pet_documents + self._sap_sam_documents

    def fold(self, num_shots: int = 3, seed: int = 42) -> list[dict[str, list[PetDocument]]]:
        """
        Generate fold for each SAP-SAM-Document with some shots from PET-Dataset.

        Parameters
        ----------
        num_shots: int = 3
            Amount of documents that must be taken as examples from PET-Dataset.
        seed: int = 42
            Seed for random generator.
            
        Returns
        -------
        list[dict[str, list[str]]]
            Returns a list of dictionaries the following structure for each SAP-SAM-Document: 
            {
                "train": list[Document from PET-Dataset as example]
                "test": list[Document from SAP-SAM-Dataset for annotation (only one)]
            }
        """
        # create a random number generator
        rng = random.Random(seed)
        # create folds
        folds = list()
        for sap_sam_document in self._sap_sam_documents:
            # get examples
            examples = random_sample_examples(
                documents=self._pet_documents, # documents that will be used as examples
                test_document_id=sap_sam_document.id, # document that must be processed
                num_examples=num_shots, # amount of examples to select
                rng=rng , # randomizer
            )
            # save fold
            folds.append(
                {
                    "train": examples,
                    "test": [sap_sam_document],
                }
            )

        return folds


def annotate_by_formatters(
        mixed_data_importer: MixedDataImporter,
        formatters: list[BaseFormattingStrategy[PetDocument]],
        *,
        chat_model: BaseChatModel,
        model_name: str,
        dry_run: bool, 
        num_shots: int = 3,
        seed: int = 42,
    ) -> list[PetDocument]:
    """
    Annotate SAP-SAM-Dateset according to formatters (Mention Detection, Entity Resolution, Relation Extraction).

    Parameters
    ----------
    mixed_data_importer: MixedDataImporter
        Importer that can provide PET examples and SAP SAM Models for annotation.
    formatters: list[BaseFormattingStrategy[PetDocument]]
        List of formatters that must be applied defining iterative or non-iterative approach.
    chat_model: BaseChatModel
        Model that will be used for annotation.
    model_name: str,
        Name of model.
    dry_run: bool
        If this run is dry.
    num_shots: int = 3
        Amount of examples for one annotation.
    seed: int = 42
        Seed for randomizer.

    Returns
    -------
    list[PetDocument]
        Documents with mentions.
    """
    # create folds
    folds = mixed_data_importer.fold(num_shots=num_shots, seed=seed)
    resulting_docs = list()
    # for each fold (SAP-SAM Document) run a model and annotate it
    for fold in tqdm(iterable=folds, desc="Annotation of documents"):
        # documents for annotation
        input_doc = fold["test"][0] # as only one element in the list
        # examples
        examples = fold["train"]
        
        # run annotation
        doc = run_iterative_document_prompt_getting_doc(
            input_document=input_doc,
            formatters=formatters,
            chat_model=chat_model,
            example_docs=examples,
            model_name=model_name,
            dry_run=dry_run,
        )

        # save document
        resulting_docs.append(doc)
    
    return resulting_docs


def get_double_assigned_tokens(pet_document: PetDocument) -> dict[int, list[str]]:
    """
    Get indices of tokens that have more than two mentions assigned.

    Parametrs
    ---------
    petDocument: PetDocument
        Document to check mentions.
    
    Returns
    -------
    dict[int, list[str]]
        Dictionary {Token_Index: list[Metion_Types assigned to this Token_Index]}. List size >= 2 for each token index.
    """
    token_mention_assignment: dict[int, list[str]] = dict()
    # for each mention get token indices and assign them mention type
    for mention in pet_document.mentions:
        for idx in mention.token_document_indices:
            if idx not in token_mention_assignment:
                token_mention_assignment[idx] = list()
            token_mention_assignment[idx].append(mention.type)
    
    # delete indices that have only oe mention type assigned
    indices = list(token_mention_assignment.keys())
    for k in indices:
        if len(token_mention_assignment[k]) < 2:
            token_mention_assignment.pop(k)
    
    # return results
    return token_mention_assignment


def remove_mention(pet_document: PetDocument, token_index: int, mention_type: str):
    """
    Remove token index from a specific mention with type in a document. Only one mention will be changed.

    Parameters
    ----------
    pet_document: PetDocument
        Document for processing.
    token_index: int
        Token index to search.
    mention_type: str
        Mention type that contains token index.
    """
    # go over all mentions
    mentions_list = pet_document.mentions
    for mention in mentions_list:
        # if there is a match regarding mention type and the token index belongs to this mention
        if (mention.type == mention_type) and token_index in mention.token_document_indices:
            # if there a mention consists of only one token, then remove a mention
            if len(mention.token_document_indices) > 1:
                # identify tokens that go before the target token index
                count_lower = [idx for idx in mention.token_document_indices if idx < token_index]
                # identify tokens that go after the target token index
                count_higher = [idx for idx in mention.token_document_indices if idx > token_index]
                # as sequence must be split, leave a longer sequence 
                new_token_document_indices = tuple(count_higher) if len(count_higher) > len(count_lower) else tuple(count_lower)
                # create a new mention and insert it instead of the current
                mentions_list.append(
                    PetMention(
                        type=mention.type, 
                        token_document_indices=new_token_document_indices
                    )
                )
            
            # process only one mention
            mentions_list.remove(mention)
            break


def double_assigned_remove(pet_document: PetDocument, double_assigned: dict[int, list[str]]):
    """
    Remove mention assignment from tokens that were assigned to several mentions.

    Parameters
    ----------
    pet_document: PetDocument
        Document that must be adjusted.
    double_assigned: dict[int, list[str]]
        All tokens with several mentions assignments and their mention types.
    """
    # go over all token indices with several mentions
    for index, mention_types in double_assigned.items():
        # while there are ambiguities, remove them
        while len(mention_types) > 1:
            mention = mention_types.pop()
            remove_mention(pet_document, index, mention)
