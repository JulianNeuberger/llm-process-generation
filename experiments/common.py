import dataclasses
import json
import os
import random
import typing

import langchain_community.callbacks
import langchain_openai
import tqdm
from langchain_core import prompts

import data
from data import base
from experiments import usage
import format

TDocument = typing.TypeVar("TDocument", bound=base.DocumentBase)


@dataclasses.dataclass
class PromptResult:
    prompt: str
    input_tokens: int
    output_tokens: int
    total_costs: float
    answer: str
    original_id: str

    def to_dict(self):
        return self.__dict__

    @staticmethod
    def from_dict(dic: typing.Dict):
        return PromptResult(
            prompt=dic["prompt"],
            input_tokens=dic["input_tokens"],
            output_tokens=dic["output_tokens"],
            total_costs=dic["total_costs"],
            answer=dic["answer"],
            original_id=dic["original_id"],
        )


@dataclasses.dataclass
class RunMeta:
    num_shots: int
    formatter: str
    model: str
    steps: typing.List[str]

    def to_dict(self):
        return self.__dict__

    @staticmethod
    def from_dict(dic: typing.Dict):
        return RunMeta(
            num_shots=dic["num_shots"],
            formatter=dic["formatter"],
            model=dic["model"],
            steps=dic["steps"],
        )


@dataclasses.dataclass
class ExperimentResult:
    meta: RunMeta
    results: typing.List[PromptResult]

    def to_dict(self):
        return {
            "meta": self.meta.to_dict(),
            "results": [r.to_dict() for r in self.results],
        }

    @staticmethod
    def from_dict(dic: typing.Dict):
        return ExperimentResult(
            meta=RunMeta.from_dict(dic["meta"]),
            results=[PromptResult.from_dict(r) for r in dic["results"]],
        )


def get_prompt(
    input_document: TDocument,
    formatter: format.BaseFormattingStrategy[TDocument],
    example_docs: typing.Iterable[TDocument],
) -> str:
    examples = [
        {"input": formatter.input(d), "steps": formatter.steps, "output": formatter.output(d)}
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
    model: langchain_openai.ChatOpenAI,
    dry_run: bool,
    num_shots: int,
) -> PromptResult:
    print(f"Running prompt for {input_document.id} ...")

    if num_shots != -1:
        assert num_shots <= len(example_docs)
        example_docs = random.sample(example_docs, num_shots)

    prompt_text = get_prompt(input_document, formatter, example_docs)
    num_input_tokens = model.get_num_tokens(prompt_text)

    if dry_run:
        print(f"Dry run for request with an estimated {num_input_tokens} tokens.")
        answer = "### DRY RUN ###"
        num_input_tokens = 0
        num_output_tokens = 0
        total_costs = 0.0
    else:
        print(f"Making request with an estimated {num_input_tokens} tokens.")
        with langchain_community.callbacks.get_openai_callback() as cb:
            res = model.invoke(prompt_text)
            num_input_tokens = cb.prompt_tokens
            num_output_tokens = cb.completion_tokens
            total_costs = usage.get_cost_for_tokens(
                model_name=model.model_name,
                num_input_tokens=num_input_tokens,
                num_output_tokens=num_output_tokens,
            )
        answer = res.content.__str__()

    return PromptResult(
        prompt=prompt_text,
        answer=answer,
        original_id=input_document.id,
        input_tokens=num_input_tokens,
        output_tokens=num_output_tokens,
        total_costs=total_costs,
    )


def run_multiple_document_prompts(
    input_documents: typing.List[TDocument],
    formatter: format.BaseFormattingStrategy[TDocument],
    example_docs: typing.List[TDocument],
    model: langchain_openai.ChatOpenAI,
    dry_run: bool,
    num_shots: int,
) -> typing.Generator[PromptResult, None, None]:
    for d in input_documents:
        yield run_single_document_prompt(
            d, formatter, example_docs, model, dry_run, num_shots
        )


def experiment(
    importer: base.BaseImporter[TDocument],
    formatter: format.BaseFormattingStrategy[TDocument],
    *,
    model_name: str,
    storage: str,
    num_shots: int,
    dry_run: bool,
    folds: typing.List[typing.Dict[str, typing.List[str]]] = None,
):
    documents = importer.do_import()
    model: langchain_openai.ChatOpenAI = langchain_openai.ChatOpenAI(
        model_name=model_name
    )

    saved_experiment_results: typing.List[ExperimentResult]
    if not os.path.isfile(storage):
        saved_experiment_results = []
    else:
        with open(storage, "r", encoding="utf8") as f:
            raw = json.load(f)
            saved_experiment_results = [ExperimentResult.from_dict(e) for e in raw]

    if folds is None:
        # experiment with no training documents
        folds = [{"train": [], "test": [d.id for d in documents]}]
        num_shots = 0

    documents_by_id = {d.id: d for d in documents}
    for fold_id, fold in tqdm.tqdm(enumerate(folds), total=len(folds)):
        if fold_id == len(saved_experiment_results):
            saved_experiment_results.append(
                ExperimentResult(
                    meta=RunMeta(
                        num_shots=num_shots,
                        model=model_name,
                        formatter=formatter.__class__.__name__,
                        steps=formatter.steps,
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

        result_iterator = run_multiple_document_prompts(
            input_documents=input_docs,
            formatter=formatter,
            num_shots=num_shots,
            model=model,
            example_docs=example_docs,
            dry_run=dry_run,
        )

        for result in result_iterator:
            current_save_fold.results.append(result)
            os.makedirs(os.path.dirname(storage), exist_ok=True)
            with open(storage, "w", encoding="utf8") as f:
                json.dump([r.to_dict() for r in saved_experiment_results], f)


def run_fold(
    fold_index: int,
    formatter: format.BaseFormattingStrategy,
    model_name: str,
    saved_results: typing.Dict,
    num_shots: int = -1,
    num_docs: int = -1,
    dry_run: bool = False,
):
    if dry_run:
        print("Running in DRY RUN mode, will not make calls to OpenAI.")
    model: langchain_openai.ChatOpenAI = langchain_openai.ChatOpenAI(
        model_name=model_name
    )
    documents = data.PetImporter("res/pet/all.new.jsonl").do_import()
    documents_by_id = {d.id: d for d in documents}

    with open("res/pet/folds.json", "r", encoding="utf8") as f:
        folds = json.load(f)
    fold = folds[fold_index]

    if num_shots == -1:
        example_doc_ids = fold["train"]
    else:
        assert num_shots < len(fold["train"])
        example_doc_ids = random.sample(fold["train"], num_shots)
    example_docs = [documents_by_id[doc_id] for doc_id in example_doc_ids]

    input_document_ids = fold["test"]
    if num_docs > -1:
        input_document_ids = input_document_ids[:num_docs]
    for input_document_id in input_document_ids:
        fold_results = saved_results["results"]
        document_ids_already_run = [r["original"] for r in fold_results]
        if input_document_id in document_ids_already_run:
            print(f"Already ran {input_document_id} in fold {fold_index}, skipping ...")
            continue

        input_document = documents_by_id[input_document_id]

        if dry_run:
            prompt_text = get_prompt(input_document, formatter, example_docs)
            yield RunResult(
                meta=RunMeta(
                    formatter=formatter.__class__.__name__,
                    num_shots=len(example_docs),
                    model=model_name,
                ),
                result=PromptResult(
                    prompt=prompt_text,
                    answer="dry run",
                    original_id=input_document_id,
                    input_tokens=0,
                    output_tokens=0,
                    total_costs=0.0,
                ),
            )
        yield run_single_document_prompt(input_document, formatter, example_docs, model)


def run_folds(
    model_name: str,
    formatter: format.BaseFormattingStrategy,
    storage: str,
    *,
    num_shots: int,
    num_folds: int,
):
    for i in tqdm.tqdm(list(range(num_folds))):
        if not os.path.isfile(storage):
            folds = []
        else:
            with open(storage, "r", encoding="utf8") as f:
                folds = json.load(f)
        if len(folds) == i:
            folds.append(
                {
                    "results": [],
                    "shots": num_shots,
                    "formatter": formatter.__class__.__name__,
                    "model": model_name,
                    "steps": formatter.steps,
                }
            )

        fold: typing.Dict = folds[i]

        for result in run_fold(
            fold_index=i,
            formatter=formatter,
            model_name=model_name,
            saved_results=fold,
            num_shots=num_shots,
            num_docs=-1,
            dry_run=False,
        ):
            if fold["formatter"] != result.meta.formatter:
                print(
                    "Formatter changed during fold! "
                    "This should not happen, still recording, so run is not lost."
                )
            if fold["shots"] != result.meta.num_shots:
                print(
                    "Number of shots changed during fold! "
                    "This should not happen, still recording, so run is not lost."
                )
            fold["results"].append(result.result.to_dict())
            with open(storage, "w", encoding="utf8") as f:
                json.dump(folds, f)


def run_single_document(
    model_name: str,
    input_document_id: str,
    formatter: format.BaseFormattingStrategy,
    storage: str,
    *,
    num_shots: int,
    fold_index: int,
):
    model: langchain_openai.ChatOpenAI = langchain_openai.ChatOpenAI(
        model_name=model_name
    )
    documents = data.PetImporter("res/data/pet/all.new.jsonl").do_import()
    documents_by_id = {d.id: d for d in documents}
    input_document = documents_by_id[input_document_id]

    with open("res/data/pet/folds.json", "r", encoding="utf8") as f:
        folds = json.load(f)
    fold = folds[fold_index]

    assert input_document_id in fold["test"]

    if num_shots == -1:
        example_doc_ids = fold["train"]
    else:
        assert num_shots < len(fold["train"])
        example_doc_ids = random.sample(fold["train"], num_shots)
    example_docs = [documents_by_id[doc_id] for doc_id in example_doc_ids]

    result = run_single_document_prompt(
        input_document=input_document,
        formatter=formatter,
        model=model,
        example_docs=example_docs,
    )

    result_dict = {
        "results": [
            {
                "prompt": result.result.prompt,
                "answer": result.result.answer,
                "original": result.result.original_id,
                "input_tokens": result.result.input_tokens,
                "output_tokens": result.result.output_tokens,
                "total_costs": result.result.total_costs,
            }
        ],
        "shots": num_shots,
        "formatter": formatter.__class__.__name__,
        "model": model_name,
        "steps": formatter.steps,
    }
    with open(storage, "w", encoding="utf8") as f:
        json.dump(result_dict, f)
