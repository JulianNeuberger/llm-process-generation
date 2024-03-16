import dataclasses
import json
import typing

import data
import eval
import experiments
import format
from format import listing

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
    experiment_result: experiments.ExperimentResult,
    importer: data.BaseImporter[TDocument],
    print_only_tags: typing.Optional[typing.List[str]],
    verbose: bool,
) -> typing.Tuple[int, ExperimentStats]:
    preds: typing.List[TDocument] = []
    truths: typing.List[TDocument] = []

    documents = importer.do_import()
    documents_by_id = {d.id: d for d in documents}

    num_parse_errors = 0
    overall_steps: typing.Optional[typing.List[str]] = None
    for result in experiment_result.results:
        predicted_doc: typing.Optional[data.DocumentBase] = None
        input_doc = documents_by_id[result.original_id]

        # If a refinement strategy is used, partial results are not considered.
        refinement_result_only = False
        if (
            listing.IterativeVanDerAaSelectiveRelationExtractionRefinementStrategy.__name__
            in result.formatters
        ):
            refinement_result_only = True

        for formatter_class_name, steps, answer, prompt, args in zip(
            result.formatters,
            result.steps,
            result.answers,
            result.prompts,
            result.formatter_args,
        ):
            if (
                refinement_result_only
                and formatter_class_name
                != listing.IterativeVanDerAaSelectiveRelationExtractionRefinementStrategy.__name__
            ):
                continue

            if overall_steps is None:
                overall_steps = steps
            assert overall_steps == steps
            formatter_class: typing.Type[format.BaseFormattingStrategy] = getattr(
                format, formatter_class_name
            )
            formatter = formatter_class(steps, **args)
            partial_prediction = formatter.parse(input_doc, answer)
            num_parse_errors += partial_prediction.num_parse_errors
            if predicted_doc is None:
                predicted_doc = partial_prediction.document
            else:
                predicted_doc = predicted_doc + partial_prediction.document

        preds.append(predicted_doc)
        truths.append(input_doc)

    assert overall_steps is not None

    stats = {}
    if "mentions" in overall_steps:
        stats_by_tag = eval.mentions_f1_stats(
            predicted_documents=preds,
            ground_truth_documents=truths,
            verbose=verbose,
            print_only_tags=print_only_tags,
        )
        stats["mentions"] = stats_by_tag
    if "entities" in overall_steps:
        stats_by_tag = eval.entity_f1_stats(
            predicted_documents=preds,
            ground_truth_documents=truths,
            verbose=verbose,
            calculate_only_tags=["Actor", "Activity Data"],
            print_only_tags=print_only_tags,
        )
        stats["entities"] = stats_by_tag
    if "relations" in overall_steps:
        stats_by_tag = eval.relation_f1_stats(
            predicted_documents=preds,
            ground_truth_documents=truths,
            verbose=verbose,
            print_only_tags=print_only_tags,
        )
        stats["relations"] = stats_by_tag
    if "constraints" in overall_steps:
        stats_by_tag = eval.constraint_f1_stats(
            predicted_documents=preds,
            ground_truth_documents=truths,
            verbose=verbose,
            print_only_tags=print_only_tags,
        )
        stats["constraints"] = stats_by_tag
    return num_parse_errors, stats


def parse_experiments(
    experiment_results: typing.List[experiments.ExperimentResult],
    importer: data.BaseImporter[TDocument],
    print_only_tags: typing.Optional[typing.List[str]],
    verbose: bool,
) -> typing.Tuple[int, typing.List[ExperimentStats]]:
    model_name = experiment_results[0].meta.model
    print(
        f"Parsing {len(experiment_results)} experiments, predicted by {model_name}..."
    )
    fold_stats: typing.List[ExperimentStats] = []

    total_parse_errors = 0
    for experiment in experiment_results:
        num_parse_errors, stats = parse_experiment(
            experiment, importer, print_only_tags, verbose
        )
        total_parse_errors += num_parse_errors
        fold_stats.append(stats)

    return total_parse_errors, fold_stats


def sum_stats(
    experiment_stats: typing.List[ExperimentStats],
) -> typing.Dict[str, typing.Dict[str, eval.Stats]]:
    total_stats_by_step: typing.Dict[str, typing.Dict[str, eval.Stats]] = {}
    for stats_by_step in experiment_stats:
        for step, stats in stats_by_step.items():
            if step not in total_stats_by_step:
                total_stats_by_step[step] = {}
            total_stats = total_stats_by_step[step]

            for tag, s in stats.items():
                if tag not in total_stats:
                    total_stats[tag] = eval.Stats(0, 0, 0)
                total_stats[tag] += s
    return total_stats_by_step


def get_num_tokens(results: typing.List[experiments.ExperimentResult]):
    return sum([r.input_tokens + r.output_tokens for e in results for r in e.results])


def get_scores(
    experiment_stats: typing.List[ExperimentStats], verbose: bool
) -> typing.Dict[str, PrintableScores]:
    total_stats_by_step = sum_stats(experiment_stats)
    if verbose:
        for step, step_stats in total_stats_by_step.items():
            print()
            print("---------------")
            print(step)
            print("---------------")
            for tag, stats in step_stats.items():
                print(f"{tag} | {stats}")
    scores_by_step = {}
    for step, stats in total_stats_by_step.items():
        f1_scores = eval.stats_to_scores(stats)
        micro_scores = eval.average_scores(stats, strategy="micro")
        macro_scores = eval.average_scores(stats, strategy="macro")
        scores_by_step[step] = PrintableScores(
            scores_by_tag=f1_scores,
            micro_averaged_scores=micro_scores,
            macro_averaged_scores=macro_scores,
        )
    return scores_by_step


def print_scores(scores: PrintableScores):
    longest_tag_length = max(
        [len(t) for t in scores.scores_by_tag.keys()] + [len("Overall (micro)")]
    )
    micro_scores = scores.micro_averaged_scores
    macro_scores = scores.macro_averaged_scores

    print(f"+-{'-' * longest_tag_length}-+-{'-' * 4}-+-{'-' * 4}-+-{'-' * 4}-+")
    print(f"| Tag{' ' * (longest_tag_length - 3)} | P    | R    | F1   |")
    print(f"+-{'-' * longest_tag_length}-+-{'-' * 4}-+-{'-' * 4}-+-{'-' * 4}-+")
    for tag, scores in scores.scores_by_tag.items():
        print(
            f"| {tag}{' ' * (longest_tag_length - len(tag))} | {scores.p:.2f} | {scores.r:.2f} | {scores.f1:.2f} |"
        )
    print(f"+-{'-' * longest_tag_length}-+-{'-' * 4}-+-{'-' * 4}-+-{'-' * 4}-+")
    print(
        f"| Overall (micro){' ' * (longest_tag_length - 15)} | {micro_scores.p:.2f} | {micro_scores.r:.2f} | {micro_scores.f1:.2f} |"
    )
    print(
        f"| Overall (macro){' ' * (longest_tag_length - 15)} | {macro_scores.p:.2f} | {macro_scores.r:.2f} | {macro_scores.f1:.2f} |"
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


def parse_file(
    result_file: str,
    only_document_ids: typing.List[str] = None,
) -> typing.List[experiments.ExperimentResult]:
    with open(result_file, "r", encoding="utf8") as f:
        contents = json.load(f)
    experiment_results = [experiments.ExperimentResult.from_dict(r) for r in contents]
    if only_document_ids is not None:
        tmp: typing.List[experiments.ExperimentResult] = []
        for e in experiment_results:
            filtered_e = experiments.ExperimentResult(e.meta, [])
            for r in e.results:
                if r.original_id in only_document_ids:
                    filtered_e.results.append(r)
            if len(filtered_e.results) > 0:
                tmp.append(filtered_e)
        experiment_results = tmp
    return experiment_results


def print_experiment_results(
    result_file: str,
    importer: data.BaseImporter[TDocument],
    only_document_ids: typing.List[str] = None,
    print_only_tags: typing.List[str] = None,
    verbose: bool = False,
):
    if print_only_tags is not None:
        print_only_tags = [t.lower() for t in print_only_tags]
    experiment_results = parse_file(result_file, only_document_ids)
    num_parse_errors, experiment_stats = parse_experiments(
        experiment_results, importer, print_only_tags, verbose
    )
    costs = parse_costs_from_experiments(experiment_results)
    print_experiment_costs(costs)

    num_shots = set([e.meta.num_shots for e in experiment_results])
    if len(num_shots):
        num_shots = list(num_shots)[0]
    print(f"Experiment used {num_shots} shots")
    unique_doc_ids = set()
    for e in experiment_results:
        for r in e.results:
            unique_doc_ids.add(r.original_id)
    print(
        f"Experimented on {len(unique_doc_ids)} unique documents, dataset has {len(importer.do_import())}."
    )
    print(list(unique_doc_ids))

    scores = get_scores(experiment_stats, verbose)
    print(
        f"Total parse errors in {len(experiment_results)} answers: {num_parse_errors}"
    )
    print_scores_by_step(scores)


def main():
    importers = {
        "pet": data.PetImporter("res/data/pet/all.new.jsonl"),
        "quishpi-re": data.VanDerAaSentenceImporter("res/data/quishpi/csv"),
        "quishpi-md": data.QuishpiImporter("res/data/quishpi", exclude_tags=["entity"]),
        "van-der-aa-re": data.VanDerAaSentenceImporter(
            "res/data/van-der-aa/datacollection.csv"
        ),
        "van-der-aa-md": data.VanDerAaImporter(
            "res/data/van-der-aa/datacollection.csv"
        ),
        "analysis": data.PetImporter("res/data/quishpi/csv/2024-03-14_13-37-53.json"),
    }

    answer_file = f"res/answers/analysis/md/baseline.json"

    importer = None
    for k, v in importers.items():
        if k in answer_file:
            importer = v
            break
    assert importer is not None

    print_experiment_results(
        answer_file,
        importer,
        # only_document_ids=["1-1_bicycle_manufacturing"],
        # print_only_tags=["action"],
        verbose=True,
    )


if __name__ == "__main__":
    main()
