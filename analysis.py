import typing

import Levenshtein
import langchain_openai
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas
import seaborn as sns

import data
import eval
import experiments
import format
from experiments import sampling, parse, common

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

plt.rcParams["text.usetex"] = True

importer = data.PetImporter("res/data/pet/all.new.jsonl")
model_name = "gpt-4-0125-preview"


def iterative_prompt():
    folds = sampling.generate_folds(importer.do_import(), num_examples=0)
    formatters = []
    tags = [
        "activity",
        "actor",
        "activity data",
        "further specification",
        "xor gateway",
        "condition specification",
        "and gateway",
    ]
    extracted_so_far = []
    for tag in tags:
        formatters.append(
            format.IterativePetMentionListingFormattingStrategy(
                ["mentions"], tag, context_tags=extracted_so_far
            )
        )

    experiments.experiment(
        importer=importer,
        formatters=formatters,
        model_name=model_name,
        storage="res/answers/analysis/md/iterative.json",
        num_shots=0,
        dry_run=False,
        folds=folds,
    )


def combined_prompt():
    folds = sampling.generate_folds(importer.do_import(), num_examples=0)
    formatters = [
        format.PetMentionListingFormattingStrategy(
            steps=["mentions"],
            prompt="pet/md/ablation/combined_prompt_no_explanation.txt",
        )
    ]

    experiments.experiment(
        importer=importer,
        formatters=formatters,
        model_name=model_name,
        storage="res/answers/analysis/md/combined.json",
        num_shots=0,
        dry_run=False,
        folds=folds,
    )


def default_prompt():
    folds = sampling.generate_folds(importer.do_import(), num_examples=0)
    formatters = [
        format.PetMentionListingFormattingStrategy(
            steps=["mentions"],
            prompt="pet/md/ablation/baseline.txt",
        )
    ]

    experiments.experiment(
        importer=importer,
        formatters=formatters,
        model_name=model_name,
        storage="res/answers/analysis/md/baseline.json",
        num_shots=0,
        dry_run=False,
        folds=folds,
    )


def no_meta_language():
    folds = sampling.generate_folds(importer.do_import(), num_examples=1)
    formatters = [
        format.PetMentionListingFormattingStrategy(
            steps=["mentions"],
            prompt="pet/md/ablation/no_meta_language.txt",
        )
    ]

    experiments.experiment(
        importer=importer,
        formatters=formatters,
        model_name=model_name,
        storage="res/answers/analysis/md/no_meta_language.json",
        num_shots=1,
        dry_run=False,
        folds=folds,
    )


def gpt_3_5():
    folds = sampling.generate_folds(importer.do_import(), num_examples=0)
    formatters = [
        format.PetMentionListingFormattingStrategy(
            steps=["mentions"],
            prompt="pet/md/ablation/baseline.txt",
        )
    ]

    experiments.experiment(
        importer=importer,
        formatters=formatters,
        model_name="gpt-3.5-turbo-0125",
        storage="res/answers/analysis/md/gpt_3_5.json",
        num_shots=0,
        dry_run=False,
        folds=folds,
    )


def no_format_examples():
    folds = sampling.generate_folds(importer.do_import(), num_examples=0)
    formatters = [
        format.PetMentionListingFormattingStrategy(
            steps=["mentions"],
            prompt="pet/md/ablation/no_format_examples.txt",
        )
    ]

    experiments.experiment(
        importer=importer,
        formatters=formatters,
        model_name=model_name,
        storage="res/answers/analysis/md/no_format_examples.json",
        num_shots=0,
        dry_run=False,
        folds=folds,
    )


def no_formatting():
    folds = sampling.generate_folds(importer.do_import(), num_examples=0)
    formatters = [
        format.PetMentionListingFormattingStrategy(
            steps=["mentions"],
            prompt="pet/md/ablation/no_formatting.txt",
        )
    ]

    experiments.experiment(
        importer=importer,
        formatters=formatters,
        model_name=model_name,
        storage="res/answers/analysis/md/no_formatting.json",
        num_shots=0,
        dry_run=False,
        folds=folds,
    )


def long_prompt():
    folds = sampling.generate_folds(importer.do_import(), num_examples=0)
    formatters = [
        format.PetMentionListingFormattingStrategy(
            steps=["mentions"],
            prompt="pet/md/ablation/long_descriptions.txt",
        )
    ]

    experiments.experiment(
        importer=importer,
        formatters=formatters,
        model_name=model_name,
        storage="res/answers/analysis/md/long_descriptions.json",
        num_shots=0,
        dry_run=False,
        folds=folds,
    )


def short_prompt():
    folds = sampling.generate_folds(importer.do_import(), num_examples=0)
    formatters = [
        format.PetMentionListingFormattingStrategy(
            steps=["mentions"],
            prompt="pet/md/ablation/short_prompt.txt",
        )
    ]

    experiments.experiment(
        importer=importer,
        formatters=formatters,
        model_name=model_name,
        storage="res/answers/analysis/md/short_explanations.json",
        num_shots=0,
        dry_run=False,
        folds=folds,
    )


def no_persona():
    folds = sampling.generate_folds(importer.do_import(), num_examples=0)
    formatters = [
        format.PetMentionListingFormattingStrategy(
            steps=["mentions"],
            prompt="pet/md/ablation/no_persona.txt",
        )
    ]

    experiments.experiment(
        importer=importer,
        formatters=formatters,
        model_name=model_name,
        storage="res/answers/analysis/md/no_persona.json",
        num_shots=0,
        dry_run=False,
        folds=folds,
    )


def no_context_manager():
    folds = sampling.generate_folds(importer.do_import(), num_examples=0)
    formatters = [
        format.PetMentionListingFormattingStrategy(
            steps=["mentions"],
            prompt="pet/md/ablation/no_context_manager.txt",
        )
    ]

    experiments.experiment(
        importer=importer,
        formatters=formatters,
        model_name=model_name,
        storage="res/answers/analysis/md/no_context_manager.json",
        num_shots=0,
        dry_run=False,
        folds=folds,
    )


def few_shots():
    max_num_shots = 10
    seed = 42
    folds = sampling.generate_folds(
        importer.do_import(), num_examples=max_num_shots, seed=seed
    )
    run_scores = []
    baseline_num_tokens: typing.Optional[int] = None
    baseline_f1: typing.Optional[float] = None
    for num_shots in range(max_num_shots + 1):
        formatters = [
            format.PetMentionListingFormattingStrategy(
                steps=["mentions"],
                prompt="pet/md/ablation/baseline.txt",
            )
        ]

        storage = f"res/answers/analysis/md/few-shots/{num_shots}.json"
        experiments.experiment(
            importer=importer,
            formatters=formatters,
            model_name=model_name,
            storage=storage,
            num_shots=num_shots,
            dry_run=False,
            folds=folds,
        )
        experiment_results = parse.parse_file(storage)

        num_tokens = sum(
            [e.input_tokens for r in experiment_results for e in r.results]
        )
        average_num_tokens_per_prompt = num_tokens / len(experiment_results)
        if baseline_num_tokens is None:
            assert num_shots == 0
            baseline_num_tokens = num_tokens

        num_parse_errors, all_experiment_stats = parse.parse_experiments(
            experiment_results, importer, print_only_tags=None, verbose=False
        )
        experiment_stats = parse.sum_stats(all_experiment_stats)

        assert len(experiment_stats) == 1, print(experiment_stats.keys())

        scores = eval.average_scores(
            list(experiment_stats.values())[0], strategy="micro"
        )

        if baseline_f1 is None:
            assert num_shots == 0
            baseline_f1 = scores.f1

        run_scores.append({"score": scores.p, "metric": "p", "num_shots": num_shots})
        run_scores.append({"score": scores.r, "metric": "r", "num_shots": num_shots})
        run_scores.append({"score": scores.f1, "metric": "f1", "num_shots": num_shots})
        run_scores.append(
            {"score": num_tokens, "metric": "tokens", "num_shots": num_shots}
        )
        run_scores.append(
            {
                "score": average_num_tokens_per_prompt,
                "metric": "average_num_tokens",
                "num_shots": num_shots,
            }
        )

        f1_improvement = scores.f1 - baseline_f1
        additional_tokens = (num_tokens - baseline_num_tokens) / len(experiment_results)

        if additional_tokens != 0:
            run_scores.append(
                {
                    "score": f1_improvement / (additional_tokens / 1000.0),
                    "metric": "token_efficiency",
                    "num_shots": num_shots,
                }
            )

    df = pandas.DataFrame.from_records(run_scores)

    sns.set_theme()

    fig = plt.figure(figsize=(8.53, 4.8))
    ax1 = fig.add_subplot(111)

    # plot precision, recall, f1
    df_as_dict = df[df.metric == "p"].to_dict(orient="list")
    ax1.plot(
        df_as_dict["num_shots"],
        df_as_dict["score"],
        color=sns.color_palette()[0],
        linestyle="dashed",
        label="$P$",
    )

    df_as_dict = df[df.metric == "r"].to_dict(orient="list")
    ax1.plot(
        df_as_dict["num_shots"],
        df_as_dict["score"],
        color=sns.color_palette()[1],
        linestyle="dashdot",
        label="$R$",
    )

    df_as_dict = df[df.metric == "f1"].to_dict(orient="list")
    ax1.plot(
        df_as_dict["num_shots"],
        df_as_dict["score"],
        color=sns.color_palette()[2],
        linestyle="solid",
        label="$F_1$",
    )
    ax1.set_ylabel("$F_1$ score")

    # plot token efficiency
    ax2 = plt.twinx()

    df_as_dict = df[df.metric == "token_efficiency"].to_dict(orient="list")
    ax2.plot(
        df_as_dict["num_shots"],
        df_as_dict["score"],
        color=sns.color_palette()[3],
        linestyle="dotted",
        label="Token Eff.",
    )
    ax2.set_ylabel("$F_1$ improvement per 1,000 tokens")

    ax1.set_yticks(np.linspace(ax1.get_ybound()[0], ax1.get_ybound()[1], 5))
    ax2.set_yticks(np.linspace(ax2.get_ybound()[0], ax2.get_ybound()[1], 5))

    ax2.grid(None)

    ax1.set_xlabel("number of examples")

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    plt.figlegend()

    plt.tight_layout(rect=(0, 0, 0.75, 1))

    plt.savefig("figures/ablation/num_shots.pdf")
    plt.savefig("figures/ablation/num_shots.png")

    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.gca()
    df_as_dict = df[df.metric == "average_num_tokens"].to_dict(orient="list")
    ax.plot(
        df_as_dict["num_shots"],
        df_as_dict["score"],
        color=sns.color_palette()[0],
        linestyle="solid",
        label="Prompt Length",
    )
    ax.set_ylabel("Prompt Length in Tokens")

    plt.figlegend()

    plt.tight_layout()

    plt.savefig("figures/ablation/prompt_length_few_shots.pdf")
    plt.savefig("figures/ablation/prompt_length_few_shots.png")


def get_vocab(strings: typing.Iterable[str]) -> typing.Dict[str, int]:
    tokens = [nltk.tokenize.word_tokenize(s) for s in strings]
    tokens = np.concatenate([np.array(t) for t in tokens], axis=0)
    unique_tokens = np.unique(tokens)
    return {t: i for i, t in enumerate(unique_tokens)}


def bag_of_words(vocab: typing.Dict[str, int], string: str) -> np.ndarray[str]:
    res = np.zeros(len(vocab))
    tokens = np.array(nltk.tokenize.word_tokenize(string))
    unique_tokens = np.unique(tokens)
    lookup_vectorized = np.vectorize(lambda token: vocab[token])
    indices = lookup_vectorized(unique_tokens)
    res[indices] = 1
    return res


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def stochasticity_minor_changes():
    num_runs = 5
    folds = sampling.generate_folds(importer.do_import(), num_examples=0, seed=42)
    run_scores = []
    scores_by_run = {}
    base_prompt = common.load_prompt_from_file("pet/md/ablation/baseline.txt")

    prompts = [
        common.load_prompt_from_file(f"pet/md/ablation/stochasticity/{i}.txt")
        for i in range(num_runs)
    ]

    vocab = get_vocab(prompts + [base_prompt])
    base_bow = bag_of_words(vocab, base_prompt)

    for i in range(num_runs):
        prompt = common.load_prompt_from_file(f"pet/md/ablation/stochasticity/{i}.txt")
        distance = Levenshtein.distance(base_prompt, prompt)
        bow = bag_of_words(vocab, prompt)
        similarity = cosine_similarity(base_bow, bow)

        formatters = [
            format.PetMentionListingFormattingStrategy(
                steps=["mentions"],
                prompt=f"pet/md/ablation/stochasticity/{i}.txt",
            )
        ]

        storage = f"res/answers/analysis/md/stochasticity/changes/{i}.json"
        experiments.experiment(
            importer=importer,
            formatters=formatters,
            model_name=model_name,
            storage=storage,
            num_shots=0,
            dry_run=False,
            folds=folds,
        )

        experiment_results = parse.parse_file(storage)
        num_parse_errors, all_experiment_stats = parse.parse_experiments(
            experiment_results, importer, print_only_tags=None, verbose=False
        )
        experiment_stats = parse.sum_stats(all_experiment_stats)

        assert len(experiment_stats) == 1, print(experiment_stats.keys())

        scores = eval.average_scores(
            list(experiment_stats.values())[0], strategy="micro"
        )
        print(f"{i}: {scores.f1}")
        scores_by_run[i] = scores.f1
        run_scores.append({"score": scores.p, "metric": "p"})
        run_scores.append({"score": scores.r, "metric": "r"})
        run_scores.append({"score": scores.f1, "metric": "f1"})
        run_scores.append({"score": distance, "metric": "levenshtein"})
        run_scores.append({"score": similarity, "metric": "cosine_similarity"})

    df = pandas.DataFrame.from_records(run_scores)
    sns.set_theme()
    plt.figure(figsize=(6.4, 4.8))
    sns.boxplot(
        data=df[df["metric"].isin(["f1", "p", "r"])], x="metric", y="score", width=0.35
    )
    plt.ylim(0, 1)
    plt.tight_layout()

    plt.savefig("figures/ablation/stochasticity_changes.pdf")
    plt.savefig("figures/ablation/stochasticity_changes.png")

    f1_df = df[df["metric"] == "f1"]["score"]
    print(
        f"Min: {f1_df.min()}, "
        f"Max: {f1_df.max()}, "
        f"Median: {f1_df.median()}, "
        f"Mean: {f1_df.mean()}, "
        f"Standard Dev.: {f1_df.std()}"
    )

    plt.figure(figsize=(4.27, 4.8))
    plt.ylim(0.45, 0.6)
    f1_scores = df[df.metric == "f1"].to_dict(orient="list")["score"]
    bow_similarities = df[df.metric == "cosine_similarity"].to_dict(orient="list")[
        "score"
    ]
    sns.scatterplot(x=bow_similarities, y=f1_scores)
    plt.hlines(
        y=0.53,
        xmin=plt.xlim()[0],
        xmax=plt.xlim()[1],
        linestyles="dotted",
        colors=sns.color_palette()[-1],
        label="Baseline Prompt $F_1$",
    )
    plt.ylabel("$F_1$")
    plt.xlabel("$S_c$")
    plt.legend()

    plt.savefig("figures/ablation/stochasticity_scatter.pdf")
    plt.savefig("figures/ablation/stochasticity_scatter.png")


def stochasticity_repeated_runs():
    num_runs = 10
    folds = sampling.generate_folds(importer.do_import(), num_examples=0, seed=42)
    run_scores = []
    scores_by_run = {}
    for i in range(num_runs):
        formatters = [
            format.PetMentionListingFormattingStrategy(
                steps=["mentions"],
                prompt="pet/md/ablation/baseline.txt",
            )
        ]

        storage = f"res/answers/analysis/md/stochasticity/repeats/{i}.json"
        experiments.experiment(
            importer=importer,
            formatters=formatters,
            model_name=model_name,
            storage=storage,
            num_shots=0,
            dry_run=False,
            folds=folds,
        )

        experiment_results = parse.parse_file(storage)
        num_parse_errors, all_experiment_stats = parse.parse_experiments(
            experiment_results, importer, print_only_tags=None, verbose=False
        )
        experiment_stats = parse.sum_stats(all_experiment_stats)

        assert len(experiment_stats) == 1, print(experiment_stats.keys())

        scores = eval.average_scores(
            list(experiment_stats.values())[0], strategy="micro"
        )
        scores_by_run[i] = scores.f1
        run_scores.append({"score": scores.p, "metric": "p"})
        run_scores.append({"score": scores.r, "metric": "r"})
        run_scores.append({"score": scores.f1, "metric": "f1"})

    df = pandas.DataFrame.from_records(run_scores)
    sns.set_theme()
    plt.figure(figsize=(4.27, 4.8))
    sns.boxplot(data=df, x="metric", y="score", width=0.35)
    plt.ylim(0, 1)
    plt.tight_layout()

    plt.savefig("figures/ablation/stochasticity.pdf")
    plt.savefig("figures/ablation/stochasticity.png")

    df = df[df["metric"] == "f1"]["score"]
    print(
        f"Min: {df.min()}, Max: {df.max()}, Median: {df.median()}, Mean: {df.mean()}, Standard Dev.: {df.std()}"
    )


def document_num_tokens():
    documents = importer.do_import()
    chat_model: langchain_openai.ChatOpenAI = langchain_openai.ChatOpenAI(
        model_name=model_name, temperature=0
    )
    lengths = [chat_model.get_num_tokens(d.text) for d in documents]
    print(
        f"Min: {min(lengths)}, Max: {max(lengths)}, Mean: {sum(lengths) / len(lengths): .1f}"
    )


if __name__ == "__main__":
    # iterative_prompt()
    default_prompt()
    no_format_examples()
    short_prompt()
    long_prompt()
    no_context_manager()
    no_persona()
    gpt_3_5()
    no_meta_language()
    # stochasticity_repeated_runs()
    # stochasticity_minor_changes()
    # few_shots()
    # document_num_tokens()
