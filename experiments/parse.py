import dataclasses
import json
import typing

import data
import experiments
from experiments import common
import format
import eval

TDocument = typing.TypeVar("TDocument", bound=data.DocumentBase)
ExperimentStats = typing.Dict[str, typing.Dict[str, eval.Stats]]


@dataclasses.dataclass
class Costs:
    num_input_tokens: int = 0
    num_output_tokens: int = 0
    total_costs: float = 0.0


@dataclasses.dataclass
class PrintableScores:
    scores_by_tag: typing.Dict[str, eval.Scores]
    micro_averaged_scores: eval.Scores
    macro_averaged_scores: eval.Scores

    def __add__(self, other):
        scores_by_tag = {}
        for k in list(self.scores_by_tag.keys()) + list(other.scores_by_tag.keys()):
            if k not in self.scores_by_tag:
                scores_by_tag[k] = other.scores_by_tag[k]
            elif k not in other.scores_by_tag:
                scores_by_tag[k] = self.scores_by_tag[k]
            else:
                scores_by_tag[k] = self.scores_by_tag[k] + other.scores_by_tag[k]

        return PrintableScores(
            scores_by_tag=scores_by_tag,
            micro_averaged_scores=self.micro_averaged_scores
            + other.micro_averaged_scores,
            macro_averaged_scores=self.macro_averaged_scores
            + other.macro_averaged_scores,
        )

    def __truediv__(self, other):
        return PrintableScores(
            scores_by_tag={k: (v / other) for k, v in self.scores_by_tag.items()},
            micro_averaged_scores=self.micro_averaged_scores / other,
            macro_averaged_scores=self.macro_averaged_scores / other,
        )


def parse_costs_from_experiments(
    experiments: typing.List[experiments.ExperimentResult],
) -> Costs:
    costs = Costs()
    for experiment in experiments:
        for result in experiment.results:
            costs.num_input_tokens += result.input_tokens
            costs.num_output_tokens += result.output_tokens
            costs.total_costs += result.total_costs
    return costs


def parse_experiment(
    experiment_result: common.ExperimentResult,
    importer: data.BaseImporter[TDocument],
    verbose: bool,
) -> ExperimentStats:
    formatter_class: typing.Type[format.BaseFormattingStrategy] = getattr(
        format, experiment_result.meta.formatter
    )
    steps = experiment_result.meta.steps

    preds: typing.List[TDocument] = []
    truths: typing.List[TDocument] = []

    documents = importer.do_import()
    documents_by_id = {d.id: d for d in documents}

    for result in experiment_result.results:
        answer = result.answer
        formatter = formatter_class(steps)
        input_doc = documents_by_id[result.original_id]
        predicted_doc = formatter.parse(input_doc, answer)
        preds.append(predicted_doc)
        truths.append(input_doc)

    stats = {}
    if "mentions" in steps:
        stats_by_tag = eval.mentions_f1_stats(
            predicted_documents=preds, ground_truth_documents=truths, verbose=verbose
        )
        stats["mentions"] = stats_by_tag
    if "entities" in steps:
        stats_by_tag = eval.entity_f1_stats(
            predicted_documents=preds,
            ground_truth_documents=truths,
            verbose=verbose,
            only_tags=["Actor", "Activity Data"],
        )
        stats["entities"] = stats_by_tag
    if "relations" in steps:
        stats_by_tag = eval.relation_f1_stats(
            predicted_documents=preds, ground_truth_documents=truths, verbose=verbose
        )
        stats["relations"] = stats_by_tag
    if "constraints" in steps:
        stats_by_tag = eval.constraint_f1_stats(
            predicted_documents=preds, ground_truth_documents=truths, verbose=verbose
        )
        stats["constraints"] = stats_by_tag
    return stats


def parse_experiments(
    experiment_results: typing.List[common.ExperimentResult],
    importer: data.BaseImporter[TDocument],
    verbose: bool,
) -> typing.List[ExperimentStats]:
    model_name = experiment_results[0].meta.model
    print(
        f"Parsing {len(experiment_results)} experiments, predicted by {model_name}..."
    )
    fold_stats: typing.List[ExperimentStats] = []

    for experiment in experiment_results:
        stats = parse_experiment(experiment, importer, verbose)
        fold_stats.append(stats)

    return fold_stats


def get_scores(
    experiment_stats: typing.List[ExperimentStats],
) -> typing.Dict[str, PrintableScores]:
    scores_by_step: typing.Dict[str, PrintableScores] = {}
    for stats_by_step in experiment_stats:
        for step, stats in stats_by_step.items():
            f1_scores = eval.stats_to_scores(stats)
            micro_scores = eval.average_scores(stats, strategy="micro")
            macro_scores = eval.average_scores(stats, strategy="macro")

            printable_scores = PrintableScores(
                scores_by_tag=f1_scores,
                micro_averaged_scores=micro_scores,
                macro_averaged_scores=macro_scores,
            )

            if step not in scores_by_step:
                scores_by_step[step] = printable_scores
            else:
                scores_by_step[step] += printable_scores
    return {k: s / len(experiment_stats) for k, s in scores_by_step.items()}


def print_scores(scores: PrintableScores):
    longest_tag_length = max(
        [len(t) for t in scores.scores_by_tag.keys()] + [len("Overall (micro)")]
    )
    micro_scores = scores.micro_averaged_scores
    macro_scores = scores.macro_averaged_scores

    print(f"+-{'-' * longest_tag_length}-+-{'-' * 4}-+-{'-' * 4}-+-{'-' * 4}-+")
    print(f"| Tag{' ' * (longest_tag_length - 3)} | F1   | R    | P    |")
    print(f"+-{'-' * longest_tag_length}-+-{'-' * 4}-+-{'-' * 4}-+-{'-' * 4}-+")
    for tag, scores in scores.scores_by_tag.items():
        print(
            f"| {tag}{' ' * (longest_tag_length - len(tag))} | {scores.f1:.2f} | {scores.r:.2f} | {scores.p:.2f} |"
        )
    print(f"+-{'-' * longest_tag_length}-+-{'-' * 4}-+-{'-' * 4}-+-{'-' * 4}-+")
    print(
        f"| Overall (micro){' ' * (longest_tag_length - 15)} | {micro_scores.f1:.2f} | {micro_scores.r:.2f} | {micro_scores.p:.2f} |"
    )
    print(
        f"| Overall (macro){' ' * (longest_tag_length - 15)} | {macro_scores.f1:.2f} | {macro_scores.r:.2f} | {macro_scores.p:.2f} |"
    )
    print(f"+-{'-' * longest_tag_length}-+-{'-' * 4}-+-{'-' * 4}-+-{'-' * 4}-+")


def print_experiment_costs(costs: Costs):
    print(
        f"Experiment sent a total of {costs.num_input_tokens} input tokens, "
        f"generated {costs.num_output_tokens} output tokens, "
        f"at a total cost of {costs.total_costs:.2f}$"
    )


def print_scores_by_step(printable_scores_by_step: typing.Dict[str, PrintableScores]):
    for step, scores in printable_scores_by_step.items():
        header_size = 30
        print()
        print("#" * header_size)
        print(f"# {step}{' ' * (header_size - len(step) - 4)} #")
        print("#" * header_size)
        print()
        print_scores(scores)


def print_experiment_results(
    result_file: str, importer: data.BaseImporter[TDocument], verbose: bool = False
):
    with open(result_file, "r", encoding="utf8") as f:
        contents = json.load(f)
    experiment_results = [experiments.ExperimentResult.from_dict(r) for r in contents]
    experiment_stats = parse_experiments(experiment_results, importer, verbose)

    costs = parse_costs_from_experiments(experiment_results)
    print_experiment_costs(costs)

    scores = get_scores(experiment_stats)
    print_scores_by_step(scores)


def main():
    print_experiment_results(
        "res/answers/vanderaa/2024-02-20_14-22-37.json",
        data.VanDerAaImporter("res/data/van-der-aa/datacollection.csv"),
        verbose=True,
    )


if __name__ == "__main__":
    main()
