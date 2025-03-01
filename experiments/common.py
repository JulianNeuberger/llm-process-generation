import json
import os
import typing

import openai
import requests
import langchain_anthropic
import langchain_community.callbacks
import langchain_openai
import langchain_mistralai
import tqdm
from langchain_community.chat_models import ChatDeepInfra
from langchain_core import prompts
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.prompt_values import PromptValue
import format
from data import base
from experiments import usage, iterative, model
from format.common import load_prompt_from_file

TDocument = typing.TypeVar("TDocument", bound=base.DocumentBase)


def get_prompt(
    formatter: format.BaseFormattingStrategy[TDocument],
    example_docs: typing.Iterable[TDocument],
) -> prompts.ChatPromptTemplate:
    examples = [
        {
            "input": formatter.input(d),
            "steps": ", ".join(formatter.steps),
            "output": formatter.output(d),
        }
        for d in example_docs
    ]

    example_template = load_prompt_from_file("example-template.txt")
    user_prompt = load_prompt_from_file("user-prompt.txt")

    example_prompt = prompts.ChatPromptTemplate.from_messages(
        [("human", user_prompt), ("ai", example_template)]
    )

    system_message = prompts.SystemMessagePromptTemplate.from_template(
        formatter.description()
    )

    few_shot_prompt = prompts.FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )

    chat_prompt = prompts.ChatPromptTemplate.from_messages(
        [system_message, few_shot_prompt, user_prompt]
    )

    return chat_prompt


def prompt_openai(
    chat_model: langchain_openai.ChatOpenAI, prompt_as_messages: PromptValue
) -> typing.Tuple[BaseMessage, float, int, int]:
    with langchain_community.callbacks.get_openai_callback() as cb:
        res = chat_model.invoke(prompt_as_messages)
        num_input_tokens = cb.prompt_tokens
        num_output_tokens = cb.completion_tokens
        total_costs = usage.get_cost_for_tokens(
            model_name=chat_model.model_name,
            num_input_tokens=num_input_tokens,
            num_output_tokens=num_output_tokens,
        )
        return res, total_costs, num_input_tokens, num_output_tokens


def run_single_document_prompt(
    input_document: TDocument,
    current_prediction: TDocument,
    formatter: format.BaseFormattingStrategy[TDocument],
    example_docs: typing.List[TDocument],
    chat_model: BaseChatModel,
    model_name: str,
    dry_run: bool,
) -> model.PromptResult:
    print(f"Running prompt for {input_document.id} ...")

    prompt = get_prompt(formatter, example_docs)

    formatted_input_document = formatter.input(current_prediction)

    prompt_as_text = prompt.format(
        input=formatted_input_document,
        steps=", ".join(formatter.steps),
    )
    prompt_as_messages = prompt.format_prompt(
        input=formatted_input_document,
        steps=", ".join(formatter.steps),
    )
    num_input_tokens = chat_model.get_num_tokens(prompt_as_text)
    num_tries = 1
    if isinstance(chat_model, langchain_openai.ChatOpenAI):
        remaining_tries = 3
        while remaining_tries > 0:
            remaining_tries -= 1
            try:
                res, total_costs, num_input_tokens, num_output_tokens = prompt_openai(
                    chat_model, prompt_as_messages
                )
                break
            except openai.InternalServerError:
                num_tries += 1
                res = BaseMessage([""], type="")
                total_costs = 0
                num_output_tokens = 0

    else:
        res = chat_model.invoke(prompt_as_messages)
        num_output_tokens = chat_model.get_num_tokens(str(res.content))
        total_costs = usage.get_cost_for_tokens(
            model_name=model_name,
            num_input_tokens=num_input_tokens,
            num_output_tokens=num_output_tokens,
        )

    if dry_run:
        print(f"Dry run for request with an estimated {num_input_tokens} tokens.")
        answer = "### DRY RUN ###"
        num_input_tokens = 0
        num_output_tokens = 0
        total_costs = 0.0
    else:
        print(f"Making request with an estimated {num_input_tokens} tokens.")

        answer = str(res.content)

    return model.PromptResult(
        prompts=[prompt_as_text],
        answers=[answer],
        formatters=[formatter.__class__.__name__],
        steps=[formatter.steps],
        original_id=input_document.id,
        input_tokens=num_input_tokens,
        output_tokens=num_output_tokens,
        total_costs=total_costs,
        formatter_args=[formatter.args],
        num_tries=[num_tries]
    )


def run_multiple_document_prompts(
    input_documents: typing.List[TDocument],
    formatter: format.BaseFormattingStrategy[TDocument],
    example_docs: typing.List[TDocument],
    chat_model: langchain_openai.ChatOpenAI,
    model_name: str,
    dry_run: bool,
) -> typing.Generator[model.PromptResult, None, None]:
    for d in input_documents:
        cur_pred = d.copy(clear=formatter.steps)
        yield run_single_document_prompt(
            d, cur_pred, formatter, example_docs, chat_model, model_name, dry_run
        )


def experiment(
    importer: base.BaseImporter[TDocument],
    formatters: typing.List[format.BaseFormattingStrategy[TDocument]],
    *,
    model_name: str,
    chat_model: BaseChatModel,
    storage: str,
    num_shots: int,
    dry_run: bool,
    folds: typing.List[typing.Dict[str, typing.List[str]]] = None,
    on_new_document: typing.Callable[[TDocument], None] = None,
    on_new_fold: typing.Callable[[int], None] = None
):
    documents = importer.do_import()

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
        if on_new_fold is not None:
            on_new_fold(fold_id)
        if fold_id == len(saved_experiment_results):
            temperature = getattr(chat_model, "temperature", -1.0)
            saved_experiment_results.append(
                model.ExperimentResult(
                    meta=model.RunMeta(
                        num_shots=num_shots,
                        model=model_name,
                        temperature=temperature,
                    ),
                    results=[],
                )
            )
        current_save_fold = saved_experiment_results[fold_id]

        documents_already_run = [r.original_id for r in current_save_fold.results]
        if len(documents_already_run) > 0:
            print(
                f"Skipping documents with ids {documents_already_run} in fold {fold_id}!"
            )

        example_docs = [documents_by_id[i] for i in fold["train"]]
        input_docs = [documents_by_id[i] for i in fold["test"]]
        input_docs = [d for d in input_docs if d.id not in documents_already_run]

        result_iterator = iterative.run_multiple_iterative_document_prompts(
            input_documents=input_docs,
            formatters=formatters,
            chat_model=chat_model,
            example_docs=example_docs,
            model_name=model_name,
            dry_run=dry_run,
            on_new_document=on_new_document
        )

        for result in result_iterator:
            current_save_fold.results.append(result)
            os.makedirs(os.path.dirname(storage), exist_ok=True)
            with open(storage, "w", encoding="utf8") as f:
                json.dump([r.to_dict() for r in saved_experiment_results], f)


def chat_model_for_name(model_name: str) -> BaseChatModel:
    if model_name.startswith("gpt-"):
        return langchain_openai.ChatOpenAI(model_name=model_name, temperature=0)
    if model_name.startswith("claude-"):
        return langchain_anthropic.ChatAnthropic(model_name=model_name, temperature=0)
    if model_name.startswith("meta-ollama-llama3.1-70b/Meta-Llama-3"):
        return ChatDeepInfra(model=model_name, temperature=0)
    if model_name.startswith("deepinfra/"):
        return ChatDeepInfra(model=model_name, temperature=0)
    if model_name.startswith("mistral"):
        return langchain_mistralai.ChatMistralAI(model=model_name, temperature=0)
    if model_name.startswith("Qwen/"):
        return langchain_openai.ChatOpenAI(
            model_name=model_name,
            temperature=0,
            openai_api_base="https://api.aimlapi.com/",
            openai_api_key=os.environ["AIML_API_KEY"],
        )
    if model_name.startswith("vllm"):
        # Name des aktuell lokal gehosteten Modells herausfinden
        model_name_url = "http://132.180.195.1:8020/v1/models"
        local_name_response = requests.get(url=model_name_url)
        data = local_name_response.json()
        local_model_name = data['data'][0]['id']
        return langchain_openai.ChatOpenAI(model_name=local_model_name, temperature=0.5,
                                           openai_api_base="http://132.180.195.1:8020/v1")

    if model_name.startswith("ollama-calme2.1-qwen2.5-72b"):
        return langchain_openai.ChatOpenAI(model_name="hf.co/mradermacher/calme-2.1-qwen2.5-72b-GGUF:Q4_K_M", temperature=0.5,
                                           openai_api_base="http://132.180.195.1:8007/v1")
    if model_name.startswith("ollama-solar-pro-preview-instruct-GGUF:Q4_K_M"):
        return langchain_openai.ChatOpenAI(model_name="hf.co/MaziyarPanahi/solar-pro-preview-instruct-GGUF:Q4_K_M", temperature=0.5,
                                           openai_api_base="http://132.180.195.1:8007/v1")
    if model_name.startswith("ollama-solar-pro-preview-instruct-GGUF:Q4_K_S"):
        return langchain_openai.ChatOpenAI(model_name="hf.co/MaziyarPanahi/solar-pro-preview-instruct-GGUF:Q4_K_S", temperature=0.5,
                                           openai_api_base="http://132.180.195.1:8007/v1")
    if model_name.startswith("ollama-shuttle-3-Q4_K_S"):
        return langchain_openai.ChatOpenAI(model_name="hf.co/mradermacher/shuttle-3-GGUF:Q4_K_S", temperature=0.5,
                                           openai_api_base="http://132.180.195.1:8007/v1")
    if model_name.startswith("ollama-llama3.3-70b-instruct-Q4_K_S"):
        return langchain_openai.ChatOpenAI(model_name="hf.co/mradermacher/Llama-3.3-70B-Instruct-GGUF:Q4_K_S", temperature=0.5,
                                           openai_api_base="http://132.180.195.1:8007/v1")
    if model_name.startswith("ollama-llama3.3-70b-instruct-Q2"):
        return langchain_openai.ChatOpenAI(model_name="hf.co/bartowski/Llama-3.3-70B-Instruct-GGUF:Q2_K", temperature=0.5,
                                           openai_api_base="http://132.180.195.1:8007/v1")
    if model_name.startswith("ollama-llama3.3-70b-instruct-Q3_K_S"):
        return langchain_openai.ChatOpenAI(model_name="hf.co/bartowski/Llama-3.3-70B-Instruct-GGUF:Q3_K_S", temperature=0.5,
                                           openai_api_base="http://132.180.195.1:8007/v1")
    if model_name.startswith("ollama-deepseek-r1-32b"):
        return langchain_openai.ChatOpenAI(model_name="deepseek-r1:32b", temperature=0.5,
                                           openai_api_base="http://132.180.195.1:8007/v1")
    if model_name.startswith("ollama-lamarck-14B"):
        return langchain_openai.ChatOpenAI(model_name="hf.co/bartowski/Lamarck-14B-v0.7-GGUF:Q2_K", temperature=0.5, openai_api_base="http://132.180.195.1:8007/v1")
    if model_name.startswith("ollama-calme3.2-instruct-78b-Q4_K_S"):
        return langchain_openai.ChatOpenAI(model_name="hf.co/mradermacher/calme-3.2-instruct-78b-GGUF:Q4_K_S", temperature=0.5, openai_api_base="http://132.180.195.1:8007/v1")
    if model_name.startswith("ollama-ultiima-32b-Q4_K_S"):
        return langchain_openai.ChatOpenAI(model_name="hf.co/mradermacher/ultiima-32B-GGUF:Q4_K_S", temperature=0.5, openai_api_base="http://132.180.195.1:8007/v1")
    if model_name.startswith("ollama-orca2-13b-Q4_K_S"):
        return langchain_openai.ChatOpenAI(model_name="hf.co/mradermacher/Orca-2-13b-GGUF:Q4_K_S", temperature=0.5, openai_api_base="http://132.180.195.1:8007/v1")
    if model_name.startswith("ollama-Qwen2.5-72B-Instruct-Q4_K_M"):
        return langchain_openai.ChatOpenAI(model_name="hf.co/bartowski/Qwen2.5-72B-Instruct-GGUF:Q4_K_M", temperature=0.5, openai_api_base="http://132.180.195.1:8007/v1")
    if model_name.startswith("ollama-ultiima-72b-Q4_K_S"):
        return langchain_openai.ChatOpenAI(model_name="hf.co/mradermacher/ultiima-72B-GGUF:Q4_K_S", temperature=0.5, openai_api_base="http://132.180.195.1:8007/v1")
    if model_name.startswith("ollama-ultiima-72b-Q3_K_S"):
        return langchain_openai.ChatOpenAI(model_name="hf.co/mradermacher/ultiima-72B-GGUF:Q3_K_S", temperature=0.5, openai_api_base="http://132.180.195.1:8007/v1")
    if model_name.startswith("ollama-L3.3-MS-Nevoria-70b-Q4_K_S"):
        return langchain_openai.ChatOpenAI(model_name="hf.co/bartowski/L3.3-MS-Nevoria-70b-GGUF:Q4_K_S", temperature=0.5, openai_api_base="http://132.180.195.1:8007/v1")
    if model_name.startswith("ollama-ultiima-32b-Q6_K"):
        return langchain_openai.ChatOpenAI(model_name="hf.co/mradermacher/ultiima-32B-GGUF:Q6_K", temperature=0.5, openai_api_base="http://132.180.195.1:8007/v1")
    if model_name.startswith("ollama-ultiima-72b-Q2_K"):
        return langchain_openai.ChatOpenAI(model_name="hf.co/mradermacher/ultiima-72B-GGUF:Q2_K", temperature=0.5, openai_api_base="http://132.180.195.1:8007/v1")
    if model_name.startswith("ollama-ultiima-32b-IQ4_XS"):
        return langchain_openai.ChatOpenAI(model_name="hf.co/mradermacher/ultiima-32B-i1-GGUF:IQ4_XS", temperature=0.5, openai_api_base="http://132.180.195.1:8007/v1")
    if model_name.startswith("ollama-calme3.2-instruct-78b-IQ4_XS"):
        return langchain_openai.ChatOpenAI(model_name="hf.co/bartowski/calme-3.2-instruct-78b-GGUF:IQ4_XS", temperature=0.5, openai_api_base="http://132.180.195.1:8007/v1")
    if model_name.startswith("ollama-mistral-large:123b-instruct-2407-q4_K_S"):
        return langchain_openai.ChatOpenAI(model_name="mistral-large:123b-instruct-2407-q4_K_S", temperature=0.5, openai_api_base="http://132.180.195.1:8007/v1")
    if model_name.startswith("ollama-nvidia_AceInstruct-72B-Q4_K_S"):
        return langchain_openai.ChatOpenAI(model_name="hf.co/bartowski/nvidia_AceInstruct-72B-GGUF:Q4_K_S", temperature=0.5, openai_api_base="http://132.180.195.1:8007/v1")
    if model_name.startswith("ollama-Lamarckvergence-14B-Q4_K_S"):
        return langchain_openai.ChatOpenAI(model_name="hf.co/mradermacher/Lamarckvergence-14B-GGUF:Q4_K_S", temperature=0.5, openai_api_base="http://132.180.195.1:8007/v1")
    if model_name.startswith("ollama-Lamarckvergence-14B-Q8_0"):
        return langchain_openai.ChatOpenAI(model_name="hf.co/mradermacher/Lamarckvergence-14B-GGUF:Q8_0", temperature=0.5, openai_api_base="http://132.180.195.1:8007/v1")
    if model_name.startswith("ollama-Lamarckvergence-14B-IQ4_XS"):
        return langchain_openai.ChatOpenAI(model_name="hf.co/mradermacher/Lamarckvergence-14B-i1-GGUF:IQ4_XS", temperature=0.5, openai_api_base="http://132.180.195.1:8007/v1")
    if model_name.startswith("ollama-solar-pro-preview-instruct-GGUF:IQ4_XS"):
        return langchain_openai.ChatOpenAI(model_name="hf.co/MaziyarPanahi/solar-pro-preview-instruct-GGUF:IQ4_XS", temperature=0.5,
                                           openai_api_base="http://132.180.195.1:8007/v1")
    if model_name.startswith("ollama-mistral-large:123b-instruct-2407-q2_K"):
        return langchain_openai.ChatOpenAI(model_name="mistral-large:123b-instruct-2407-q2_K", temperature=0.5, openai_api_base="http://132.180.195.1:8007/v1")
    if model_name.startswith("ollama-Lamarckvergence-14B-IQ4_NL"):
        return langchain_openai.ChatOpenAI(model_name="hf.co/mradermacher/Lamarckvergence-14B-i1-GGUF:IQ4_NL", temperature=0.5, openai_api_base="http://132.180.195.1:8007/v1")
    if model_name.startswith("ollama-c4ai-command-r-plus-08-2024-Q5_K_M"):
        return langchain_openai.ChatOpenAI(model_name="command-r-plus:104b-08-2024-q5_K_M", temperature=0.5, openai_api_base="http://132.180.195.1:8007/v1")
    raise ValueError(f'Unknown model with name "{model_name}"')
