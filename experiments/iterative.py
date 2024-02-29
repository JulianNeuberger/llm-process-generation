import typing

import langchain_openai

from data import base
import format
from experiments import model, common

TDocument = typing.TypeVar("TDocument", bound=base.DocumentBase)


def run_iterative_document_prompt(
    input_document: TDocument,
    formatters: typing.List[format.BaseFormattingStrategy[TDocument]],
    example_docs: typing.List[TDocument],
    chat_model: langchain_openai.ChatOpenAI,
    dry_run: bool,
    num_shots: int,
) -> model.PromptResult:
    merged_result: typing.Optional[model.PromptResult] = None
    for formatter in formatters:
        result = common.run_single_document_prompt(
            input_document, formatter, example_docs, chat_model, dry_run, num_shots
        )
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
    chat_model: langchain_openai.ChatOpenAI,
    dry_run: bool,
    num_shots: int,
) -> typing.Generator[model.PromptResult, None, None]:
    for d in input_documents:
        yield run_iterative_document_prompt(
            d, formatters, example_docs, chat_model, dry_run, num_shots
        )
