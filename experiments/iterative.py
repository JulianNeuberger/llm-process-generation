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
    chat_model: langchain_openai.ChatOpenAI,
    dry_run: bool,
) -> typing.Generator[model.PromptResult, None, None]:
    for d in input_documents:
        yield run_iterative_document_prompt(
            d, formatters, example_docs, chat_model, dry_run
        )
