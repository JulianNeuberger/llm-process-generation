import json
import os
import random
import typing

import langchain_community.callbacks
import langchain_openai
import tqdm
from langchain_core import prompts

import format
from data import base
from experiments import usage, iterative, model

TDocument = typing.TypeVar("TDocument", bound=base.DocumentBase)


def get_prompt(
    input_document: TDocument,
    formatter: format.BaseFormattingStrategy[TDocument],
    example_docs: typing.Iterable[TDocument],
) -> str:
    examples = [
        {
            "input": formatter.input(d),
            "steps": formatter.steps,
            "output": formatter.output(d),
        }
        for d in example_docs
    ]

    example_template = """User: Please retrieve all {steps} from the following text: 
        Text: {input}
        Elements: {steps}

        AI: {output}"""

    example_prompt = prompts.PromptTemplate(
        input_variables=["input", "output", "steps"],
        template=example_template,
    )

    system_message = formatter.description()

    user_prompt = (
        "User: Please retrieve all {steps} from the following text: \n"
        "Text: {input}\n"
        "Elements: {steps}\n"
        "AI:"
    )

    few_shot_prompt = prompts.FewShotPromptTemplate(
        input_variables=["input", "steps"],
        example_separator="\n\n",
        example_prompt=example_prompt,
        examples=examples,
        prefix=system_message,
        suffix=user_prompt,
    )
    formatted_input_document = formatter.input(input_document)
    return few_shot_prompt.format(
        input=formatted_input_document,
        steps=", ".join(formatter.steps),
    )


def run_single_document_prompt(
    input_document: TDocument,
    formatter: format.BaseFormattingStrategy[TDocument],
    example_docs: typing.List[TDocument],
    chat_model: langchain_openai.ChatOpenAI,
    dry_run: bool,
    num_shots: int,
) -> model.PromptResult:
    print(f"Running prompt for {input_document.id} ...")

    if num_shots != -1:
        assert num_shots <= len(example_docs)
        example_docs = random.sample(example_docs, num_shots)

    prompt_text = get_prompt(input_document, formatter, example_docs)
    num_input_tokens = chat_model.get_num_tokens(prompt_text)

    if dry_run:
        print(f"Dry run for request with an estimated {num_input_tokens} tokens.")
        answer = "### DRY RUN ###"
        num_input_tokens = 0
        num_output_tokens = 0
        total_costs = 0.0
    else:
        print(f"Making request with an estimated {num_input_tokens} tokens.")
        with langchain_community.callbacks.get_openai_callback() as cb:
            res = chat_model.invoke(prompt_text)
            num_input_tokens = cb.prompt_tokens
            num_output_tokens = cb.completion_tokens
            total_costs = usage.get_cost_for_tokens(
                model_name=chat_model.model_name,
                num_input_tokens=num_input_tokens,
                num_output_tokens=num_output_tokens,
            )
        answer = res.content.__str__()

    return model.PromptResult(
        prompts=[prompt_text],
        answers=[answer],
        formatters=[formatter.__class__.__name__],
        steps=[formatter.steps],
        original_id=input_document.id,
        input_tokens=num_input_tokens,
        output_tokens=num_output_tokens,
        total_costs=total_costs,
    )


def run_multiple_document_prompts(
    input_documents: typing.List[TDocument],
    formatter: format.BaseFormattingStrategy[TDocument],
    example_docs: typing.List[TDocument],
    chat_model: langchain_openai.ChatOpenAI,
    dry_run: bool,
    num_shots: int,
) -> typing.Generator[model.PromptResult, None, None]:
    for d in input_documents:
        yield run_single_document_prompt(
            d, formatter, example_docs, chat_model, dry_run, num_shots
        )


def experiment(
    importer: base.BaseImporter[TDocument],
    formatters: typing.List[format.BaseFormattingStrategy[TDocument]],
    *,
    model_name: str,
    storage: str,
    num_shots: int,
    dry_run: bool,
    folds: typing.List[typing.Dict[str, typing.List[str]]] = None,
):
    documents = importer.do_import()
    chat_model: langchain_openai.ChatOpenAI = langchain_openai.ChatOpenAI(
        model_name=model_name, temperature=0
    )

    saved_experiment_results: typing.List[model.ExperimentResult]
    if not os.path.isfile(storage):
        saved_experiment_results = []
    else:
        with open(storage, "r", encoding="utf8") as f:
            raw = json.load(f)
            saved_experiment_results = [
                model.ExperimentResult.from_dict(e) for e in raw
            ]

    if folds is None:
        # experiment with no training documents
        folds = [{"train": [], "test": [d.id for d in documents]}]
        num_shots = 0

    documents_by_id = {d.id: d for d in documents}
    for fold_id, fold in tqdm.tqdm(enumerate(folds), total=len(folds)):
        if fold_id == len(saved_experiment_results):
            saved_experiment_results.append(
                model.ExperimentResult(
                    meta=model.RunMeta(
                        num_shots=num_shots,
                        model=model_name,
                        temperature=chat_model.temperature,
                    ),
                    results=[],
                )
            )
        current_save_fold = saved_experiment_results[fold_id]

        documents_already_run = [r.original_id for r in current_save_fold.results]
        print(f"Skipping documents with ids {documents_already_run} in fold {fold_id}!")

        example_docs = [documents_by_id[i] for i in fold["train"]]
        input_docs = [documents_by_id[i] for i in fold["test"]]
        input_docs = [d for d in input_docs if d.id not in documents_already_run]

        result_iterator = iterative.run_multiple_iterative_document_prompts(
            input_documents=input_docs,
            formatters=formatters,
            num_shots=num_shots,
            chat_model=chat_model,
            example_docs=example_docs,
            dry_run=dry_run,
        )

        for result in result_iterator:
            print(result.to_dict().keys())
            current_save_fold.results.append(result)
            os.makedirs(os.path.dirname(storage), exist_ok=True)
            with open(storage, "w", encoding="utf8") as f:
                json.dump([r.to_dict() for r in saved_experiment_results], f)
