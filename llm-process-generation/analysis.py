import typing

import Levenshtein
import langchain_openai
import matplotlib
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas
import seaborn as sns
import tqdm

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

# task = "re"
task = "md"


def _set_theme():
    sns.set_theme(
        rc={
            "figure.autolayout": False,
            "font.family": ["Computer Modern", "CMU Serif", "cmu", "serif"],
            "font.serif": ["Computer Modern", "CMU Serif", "cmu"],
            #'text.usetex': True
        }
    )
    matplotlib.rcParams.update(
        {
            "figure.autolayout": False,
            "font.family": ["Computer Modern", "CMU Serif", "cmu", "serif"],
            "font.serif": ["Computer Modern", "CMU Serif", "cmu"],
            #'text.usetex': True
        }
    )
    sns.set_style(
        rc={
            "font.family": ["Computer Modern", "CMU Serif", "cmu", "serif"],
            "font.serif": ["Computer Modern", "CMU Serif", "cmu"],
            #'text.usetex': True
        }
    )
    sns.set(font="Computer Modern", font_scale=1.25)


def iterative_prompt():
    folds = sampling.generate_folds(importer.do_import(), num_examples=0)

    if task == "re":
        formatters = [
            format.PetIterativeRelationListingFormattingStrategy(
                ["relations"],
                "pet/re/iterative/same_gateway.txt",
                only_tags=["same gateway"],
            ),
            format.PetIterativeRelationListingFormattingStrategy(
                ["relations"],
                "pet/re/iterative/flow.txt",
                only_tags=["flow"],
            ),
            format.PetIterativeRelationListingFormattingStrategy(
                ["relations"],
                "pet/re/iterative/remaining.txt",
                only_tags=[
                    "uses",
                    "actor performer",
                    "actor recipient",
                    "further specification",
                ],
            ),
        ]
    else:
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
        formatters = []
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
        storage=f"res/answers/analysis/{task}/iterative.json",
        num_shots=0,
        dry_run=False,
        folds=folds,
    )


def default_prompt():
    folds = sampling.generate_folds(importer.do_import(), num_examples=0)
    if task == "md":
        formatters = [
            format.PetMentionListingFormattingStrategy(
                steps=["mentions"],
                prompt=f"pet/{task}/ablation/baseline.txt",
            )
        ]
    else:
        formatters = [
            format.PetRelationListingFormattingStrategy(
                steps=["relations"],
                prompt=f"pet/{task}/ablation/baseline.txt",
            )
        ]

    experiments.experiment(
        importer=importer,
        formatters=formatters,
        model_name=model_name,
        storage=f"res/answers/analysis/{task}/baseline.json",
        num_shots=0,
        dry_run=False,
        folds=folds,
    )


def gpt_3_5():
    folds = sampling.generate_folds(importer.do_import(), num_examples=0)
    if task == "md":
        formatters = [
            format.PetMentionListingFormattingStrategy(
                steps=["mentions"],
                prompt=f"pet/{task}/ablation/baseline.txt",
            )
        ]
    else:
        formatters = [
            format.PetRelationListingFormattingStrategy(
                steps=["relations"],
                prompt=f"pet/{task}/ablation/baseline.txt",
            )
        ]

    experiments.experiment(
        importer=importer,
        formatters=formatters,
        model_name="gpt-3.5-turbo-0125",
        storage=f"res/answers/analysis/{task}/gpt_3_5.json",
        num_shots=0,
        dry_run=False,
        folds=folds,
    )


def run_ablation(prompt_name: str):
    folds = sampling.generate_folds(importer.do_import(), num_examples=0)

    if task == "md":
        formatters = [
            format.PetMentionListingFormattingStrategy(
                steps=["mentions"],
                prompt=f"pet/{task}/ablation/{prompt_name}.txt",
            )
        ]
    else:
        formatters = [
            format.PetRelationListingFormattingStrategy(
                steps=["relations"],
                prompt=f"pet/{task}/ablation/{prompt_name}.txt",
            )
        ]

    experiments.experiment(
        importer=importer,
        formatters=formatters,
        model_name=model_name,
        storage=f"res/answers/analysis/{task}/{prompt_name}.json",
        num_shots=0,
        dry_run=False,
        folds=folds,
    )


def few_shots():
    if task == "re":
        return

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
                prompt=f"pet/{task}/ablation/baseline.txt",
            )
        ]

        storage = f"res/answers/analysis/{task}/few-shots/{num_shots}.json"
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

    fig = plt.figure(figsize=(8.53, 4.8))
    ax1 = fig.add_subplot(111)

    # plot precision, recall, f1
    df_as_dict = df[df.metric == "p"].to_dict(orient="list")
    ax1.plot(
        df_as_dict["num_shots"],
        df_as_dict["score"],
        color=sns.color_palette()[0],
        linestyle="dashed",
        linewidth=2,
        label="$P$",
    )

    df_as_dict = df[df.metric == "r"].to_dict(orient="list")
    ax1.plot(
        df_as_dict["num_shots"],
        df_as_dict["score"],
        color=sns.color_palette()[1],
        linestyle="dashdot",
        linewidth=2,
        label="$R$",
    )

    df_as_dict = df[df.metric == "f1"].to_dict(orient="list")
    ax1.plot(
        df_as_dict["num_shots"],
        df_as_dict["score"],
        color=sns.color_palette()[2],
        linestyle="solid",
        linewidth=2,
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
        linewidth=2,
        label="Token Eff.",
    )
    ax2.set_ylabel("$\\Delta F_1$ / 1,000 tokens")

    ax1.set_yticks(np.linspace(ax1.get_ybound()[0], ax1.get_ybound()[1], 5))
    ax2.set_yticks(np.linspace(ax2.get_ybound()[0], ax2.get_ybound()[1], 5))

    ax2.grid(None)

    ax1.set_xlabel("number of examples")

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    plt.figlegend()

    plt.tight_layout(rect=(0, 0, 0.75, 1))

    plt.savefig(f"figures/ablation/{task}/num_shots.pdf")
    plt.savefig(f"figures/ablation/{task}/num_shots.png")

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

    plt.savefig(f"figures/ablation/{task}/prompt_length_few_shots.pdf")
    plt.savefig(f"figures/ablation/{task}/prompt_length_few_shots.png")


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
    base_prompt = common.load_prompt_from_file(f"pet/{task}/ablation/baseline.txt")

    prompts = [
        common.load_prompt_from_file(f"pet/{task}/ablation/stochasticity/{i}.txt")
        for i in range(num_runs)
    ]

    vocab = get_vocab(prompts + [base_prompt])
    base_bow = bag_of_words(vocab, base_prompt)

    for i in range(num_runs):
        prompt = common.load_prompt_from_file(
            f"pet/{task}/ablation/stochasticity/{i}.txt"
        )
        distance = Levenshtein.distance(base_prompt, prompt)
        bow = bag_of_words(vocab, prompt)
        similarity = cosine_similarity(base_bow, bow)

        if task == "md":
            formatters = [
                format.PetMentionListingFormattingStrategy(
                    steps=["mentions"],
                    prompt=f"pet/{task}/ablation/stochasticity/{i}.txt",
                )
            ]
        else:
            formatters = [
                format.PetRelationListingFormattingStrategy(
                    steps=["relations"],
                    prompt=f"pet/{task}/ablation/stochasticity/{i}.txt",
                )
            ]

        storage = f"res/answers/analysis/{task}/stochasticity/changes/{i}.json"
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
    plt.figure(figsize=(6.4, 4.8))
    sns.boxplot(
        data=df[df["metric"].isin(["f1", "p", "r"])], x="metric", y="score", width=0.35
    )
    plt.ylim(0, 1)
    plt.tight_layout()

    plt.savefig(f"figures/ablation/{task}/stochasticity_changes.pdf")
    plt.savefig(f"figures/ablation/{task}/stochasticity_changes.png")

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

    plt.savefig(f"figures/ablation/{task}/stochasticity_scatter.pdf")
    plt.savefig(f"figures/ablation/{task}/stochasticity_scatter.png")


def stochasticity_repeated_runs():
    num_runs = 10
    folds = sampling.generate_folds(importer.do_import(), num_examples=0, seed=42)
    run_scores = []
    scores_by_run = {}
    for i in range(num_runs):
        if task == "md":
            formatters = [
                format.PetMentionListingFormattingStrategy(
                    steps=["mentions"],
                    prompt=f"pet/{task}/ablation/baseline.txt",
                )
            ]
        else:
            formatters = [
                format.PetRelationListingFormattingStrategy(
                    steps=["relations"],
                    prompt=f"pet/{task}/ablation/baseline.txt",
                )
            ]

        storage = f"res/answers/analysis/{task}/stochasticity/repeats/{i}.json"
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
    plt.figure(figsize=(4.27, 4.8))
    sns.boxplot(data=df, x="metric", y="score", width=0.35)
    plt.ylim(0, 1)
    plt.tight_layout()

    plt.savefig(f"figures/ablation/{task}/stochasticity.pdf")
    plt.savefig(f"figures/ablation/{task}/stochasticity.png")

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


def bar_plot():
    def get_statistics_from_experiment(
        exp_file_path: str,
    ):
        results = parse.parse_file(exp_file_path)
        print(f"NUM RESULTS: {len(results)}")
        errors, stats = parse.parse_experiments(
            results, importer, print_only_tags=None, verbose=False
        )
        scores = parse.get_scores(stats, verbose=False)
        num_tokens = parse.get_num_tokens(results) / len(results)
        assert len(scores) == 1
        return list(scores.values())[0].micro_averaged_scores.f1, errors, num_tokens

    def draw_labels(xs, ys, offset=1e-4):
        for _x, _y in zip(xs, ys):
            ax.text(
                _y,  # + math.copysign(_y, offset),
                _x,
                f"{_y:.2f}",
                horizontalalignment="left" if _y > 0 else "right",
                verticalalignment="center",
                fontname="CMS",
            )

    experiment_names = {
        # "": "Iterative",
        "no_format_examples": "No Format Example",
        # "": "No Formatting",
        "short_descriptions": "Short Prompt",
        # "long_descriptions": "Long Prompt",
        "gpt_3_5": "GPT 3.5",
        "no_context_manager": "No Context Manager",
        "no_persona": "No Persona",
        "no_meta_language": "No Meta Language",
        "no_cot": "No Chain of Thoughts",
        "no_disambiguation": "No Disambiguation",
        "no_explanations": "No Explanation Generation",
        "no_facts": "No Fact Generation",
    }
    plot_experiments = [
        "no_format_examples",
        "short_descriptions",
        "gpt_3_5",
        "no_meta_language",
        "no_explanations",
    ]
    md_baseline, md_baseline_errors, md_baseline_tokens = (
        get_statistics_from_experiment(f"res/answers/analysis/md/baseline.json")
    )
    md_values = []
    md_errors = []
    md_tokens = []
    re_baseline, re_baseline_errors, re_baseline_tokens = (
        get_statistics_from_experiment(f"res/answers/analysis/re/baseline.json")
    )
    re_values = []
    re_errors = []
    re_tokens = []
    for exp in tqdm.tqdm(list(experiment_names.keys())):
        md_value, num_md_errors, md_num_tokens = get_statistics_from_experiment(
            f"res/answers/analysis/md/{exp}.json"
        )
        md_errors.append(num_md_errors)
        md_values.append(md_value - md_baseline)
        md_tokens.append(md_num_tokens - md_baseline_tokens)
        try:
            re_value, num_re_errors, re_num_tokens = get_statistics_from_experiment(
                f"res/answers/analysis/re/{exp}.json"
            )
            re_values.append(re_value - re_baseline)
            re_errors.append(num_re_errors)
            re_tokens.append(re_num_tokens - re_baseline_tokens)
        except FileNotFoundError:
            print(f"missing file 'res/answers/analysis/re/{exp}.json'")
            re_values.append(-10)
            re_errors.append(-1)
            re_tokens.append(-1)

    fig, ax = plt.subplots()

    data_points = {"experiment_names": [], "md_values": [], "re_values": []}

    for exp_name, md_value, re_value in zip(
        experiment_names.keys(), md_values, re_values
    ):
        if exp_name not in plot_experiments:
            continue
        data_points["experiment_names"].append(exp_name)
        data_points["md_values"].append(md_value)
        data_points["re_values"].append(re_value)

    y_pos = np.arange(len(data_points["experiment_names"]))
    height = 0.8
    y_pos_md = y_pos - height / 4
    y_pos_re = y_pos + height / 4
    ax.barh(
        y_pos_md, data_points["md_values"], height=height / 2, hatch="///", label="MD"
    )
    ax.barh(
        y_pos_re,
        data_points["re_values"],
        height=height / 2,
        hatch="\\\\\\",
        label="RE",
    )
    ax.set_yticks(y_pos, [experiment_names[k] for k in data_points["experiment_names"]])
    ax.invert_yaxis()

    draw_labels(y_pos_md, data_points["md_values"])
    draw_labels(y_pos_re, data_points["re_values"])

    bottom, top = ax.get_xlim()
    ax.set_xlim(bottom * 1.1, top * 1.1)
    ax.set_xlabel("$\\Delta F_1$")

    ax.legend()

    plt.tight_layout()

    plt.savefig("figures/ablation/bar.pdf")
    plt.savefig("figures/ablation/bar.png")

    max_experiment_name = max([len(n) for n in experiment_names])
    print(
        f"{'name':>{max_experiment_name}} | MD Rel. | MD Abs. | MD Err. | RE Rel. | RE Abs. | RE Err. "
    )
    print(
        f"{'-' * max_experiment_name}-+---------+---------+---------+---------+---------+---------"
    )
    print(
        f"{'Baseline':>{max_experiment_name}} |   ---   | {md_baseline:+.2f}   | {md_baseline_errors:>7} |   ---   | {re_baseline:+.2f}   | {re_baseline_errors:>7}"
    )
    for (
        exp_name,
        md_diff,
        md_errors,
        md_num_tokens,
        re_diff,
        re_errors,
        re_num_tokens,
    ) in zip(
        experiment_names,
        md_values,
        md_errors,
        md_tokens,
        re_values,
        re_errors,
        re_tokens,
    ):
        absolute_md = md_baseline + md_diff
        absolute_re = re_baseline + re_diff
        print(
            f"{exp_name:>{max_experiment_name}} | {md_diff:+.2f}   | {absolute_md:.2f}    | {md_errors:>7} | {re_diff:+.2f}   | {absolute_re:.2f}    | {re_errors:>7}"
        )


if __name__ == "__main__":
    _set_theme()

    # default_prompt()
    #
    # run_ablation("no_context_manager")
    # run_ablation("no_cot")
    # run_ablation("no_disambiguation")
    # run_ablation("no_explanations")
    # run_ablation("no_facts")
    # run_ablation("no_format_examples")
    # run_ablation("no_meta_language")
    # run_ablation("no_persona")
    # run_ablation("short_descriptions")
    #
    # gpt_3_5()

    # stochasticity_repeated_runs()
    # stochasticity_minor_changes()
    # few_shots()
    # document_num_tokens()

    bar_plot()
