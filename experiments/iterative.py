import typing

from langchain_core.language_models import BaseChatModel

import format
from data import base
from experiments import model, common
from annotate_sap_sam.time_checker import TimeChecker

TDocument = typing.TypeVar("TDocument", bound=base.DocumentBase)

@TimeChecker
def run_iterative_document_prompt_getting_doc(
    input_document: TDocument,
    formatters: typing.List[format.BaseFormattingStrategy[TDocument]],
    example_docs: typing.List[TDocument],
    chat_model: BaseChatModel,
    model_name: str,
    dry_run: bool,
) -> TDocument:
    """
    Run prompt and get an updated copy of the target document.

    Parameters
    ----------
    input_document: TDocument
        Input document to annotate. 
    formatters: typing.List[format.BaseFormattingStrategy[TDocument]]
        Formatters that specify annotation type.
    example_docs: typing.List[TDocument]
        Documents provided as examples.
    chat_model: BaseChatModel
        AI Model that will be used for annotation.
    model_name: str
        Name of AI Model.
    dry_run: bool
        If run is dry.

    Returns
    -------
    TDocument
        An updated copy of input_document after annotation. 
    """
    # copy the original document
    cur_doc: TDocument = input_document.copy(formatters[0].steps)
    # apply each formatter
    for i, formatter in enumerate(formatters):
        # run a single document prompt
        result = common.run_single_document_prompt(
            input_document,
            cur_doc,
            formatter,
            example_docs,
            chat_model,
            model_name,
            dry_run,
        )

        # parse answers and add them to the document
        for answer in result.answers:
            parsed = formatter.parse(input_document, answer)
            if cur_doc is None:
                cur_doc = parsed.document
            else:
                cur_doc += parsed.document
        
    return cur_doc


def run_iterative_document_prompt(
    input_document: TDocument,
    formatters: typing.List[format.BaseFormattingStrategy[TDocument]],
    example_docs: typing.List[TDocument],
    chat_model: BaseChatModel,
    model_name: str,
    dry_run: bool,
) -> model.PromptResult:
    merged_result: typing.Optional[model.PromptResult] = None
    cur_doc: TDocument = input_document.copy(formatters[0].steps)
    for i, formatter in enumerate(formatters):
        if len(formatters) > 1:
            print(
                f"Running partial prompt {i + 1}/{len(formatters)} ({formatter.__class__.__name__}) for document {input_document.id}"
            )
        result = common.run_single_document_prompt(
            input_document,
            cur_doc,
            formatter,
            example_docs,
            chat_model,
            model_name,
            dry_run,
        )

        for answer in result.answers:
            parsed = formatter.parse(input_document, answer)
            if cur_doc is None:
                cur_doc = parsed.document
            else:
                cur_doc += parsed.document
        if merged_result is None:
            merged_result = result
        else:
            merged_result = merged_result + result
    assert merged_result is not None
    return merged_result


def run_multiple_iterative_document_prompts(
    input_documents: typing.List[TDocument],
    formatters: typing.List[format.BaseFormattingStrategy[TDocument]],
    example_docs: typing.List[TDocument],
    chat_model: BaseChatModel,
    model_name: str,
    dry_run: bool,
) -> typing.Generator[model.PromptResult, None, None]:
    for d in input_documents:
        yield run_iterative_document_prompt(
            d, formatters, example_docs, chat_model, model_name, dry_run
        )
