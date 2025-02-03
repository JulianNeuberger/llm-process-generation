import datetime
import os
import statistics
import pandas as pd
import nltk
from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel

import data
import experiments
import format
from experiments import sampling, power
from experiments.parse import parse_experiments, get_scores

if __name__ == "__main__":
    def run_experiments(experiment_type: str, directory_path: str, model_name: str, iter: int):

        if experiment_type == "pet_md":
            # copy of pet_md
            load_dotenv()

            # Load sentence tokenizer if necessary
            try:
                nltk.data.find("tokenizers/punkt")
            except LookupError:
                nltk.download("punkt")

            num_shots = 3

            storage = directory_path + f"answers/iteration{iter}.json"

            # formatter = format.PetMentionListingFormattingStrategy(["mentions"])
            importer = data.PetImporter("res/data/pet/all.new.jsonl")
            # train_docs = [d.id for d in importer.do_import() if d.id != "doc-6.1"]
            # folds = [{"train": train_docs, "test": ["doc-6.1"]}]
            folds = sampling.generate_folds(
                documents=importer.do_import(),
                num_examples=num_shots,
                strategy="similarity",
                seed=42,
            )

            # formatters = [
            #     format.PetActivityListingFormattingStrategy(["mentions"]),
            #     format.PetActorListingFormattingStrategy(["mentions"]),
            #     format.PetDataListingFormattingStrategy(["mentions"]),
            #     format.PetFurtherListingFormattingStrategy(["mentions"]),
            #     format.PetXorListingFormattingStrategy(["mentions"]),
            #     format.PetConditionListingFormattingStrategy(["mentions"]),
            #     format.PetAndListingFormattingStrategy(["mentions"]),
            # ]

            formatters = [
                format.IterativePetMentionListingFormattingStrategy(
                    ["mentions"],
                    "activity",
                    context_tags=[],
                    # prompt="pet/md/iterative/with_explanation/activity.txt",
                ),
                format.IterativePetMentionListingFormattingStrategy(
                    ["mentions"],
                    "actor",
                    context_tags=["activity"],
                    # prompt="pet/md/iterative/with_explanation/actor.txt",
                ),
                format.IterativePetMentionListingFormattingStrategy(
                    ["mentions"],
                    "activity data",
                    context_tags=["activity", "actor"],
                    # prompt="pet/md/iterative/with_explanation/activity_data.txt",
                ),
                format.IterativePetMentionListingFormattingStrategy(
                    ["mentions"],
                    "further specification",
                    context_tags=["activity", "actor", "activity data"],
                    # prompt="pet/md/iterative/with_explanation/further_specification.txt",
                ),
                format.IterativePetMentionListingFormattingStrategy(
                    ["mentions"],
                    "xor gateway",
                    context_tags=[
                        "activity",
                        "actor",
                        "activity data",
                        "further specification",
                    ],
                    # prompt="pet/md/iterative/with_explanation/xor_gateway.txt",
                ),
                format.IterativePetMentionListingFormattingStrategy(
                    ["mentions"],
                    "condition specification",
                    context_tags=[
                        "activity",
                        "actor",
                        "activity data",
                        "further specification",
                        "xor gateway",
                    ],
                    # prompt="pet/md/iterative/with_explanation/condition_specification.txt",
                ),
                format.IterativePetMentionListingFormattingStrategy(
                    ["mentions"],
                    "and gateway",
                    context_tags=[
                        "activity",
                        "actor",
                        "activity data",
                        "further specification",
                        "xor gateway",
                        "condition specification",
                    ],
                    # prompt="pet/md/iterative/with_explanation/and_gateway.txt",
                ),
            ]

            formatters = [
                format.PetMentionListingFormattingStrategy(
                    steps=["mentions"],
                    only_tags=None,
                    generate_descriptions=False,
                    prompt="pet/md/unified.txt",
                )
            ]

            print("Using folds:")
            print("------------")
            for fold in folds:
                print(fold)
            print("------------")

            chat_model: BaseChatModel = experiments.chat_model_for_name(model_name)
            print(f"Using model: {chat_model.name}")

            experiments.experiment(
                importer=importer,
                formatters=formatters,
                model_name=model_name,
                chat_model=chat_model,
                storage=storage,
                num_shots=num_shots,
                dry_run=False,
                folds=folds,
                on_new_document=power_logger.set_current_document,
                on_new_fold=power_logger.set_current_fold_id
            )

        elif experiment_type == "pet_re":
            # copy of pet_re
            load_dotenv()

            # Load sentence tokenizer if necessary
            try:
                nltk.data.find("tokenizers/punkt")
            except LookupError:
                nltk.download("punkt")

            num_shots = 1

            storage = directory_path + f"answers/iteration{iter}.json"

            # formatter = format.PetMentionListingFormattingStrategy(["mentions"])
            importer = data.PetImporter("res/data/pet/all.new.jsonl")
            # folds = [
            #     {
            #         "train": [d.id for d in importer.do_import() if d.id != "doc-6.1"],
            #         "test": ["doc-6.1"],
            #     }
            # ]
            folds = sampling.generate_folds(
                importer.do_import(), num_shots, strategy="similarity"
            )

            # formatters = [format.PetRelationListingFormattingStrategy(steps=["relations"])]
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

            print("Using folds:")
            print("------------")
            for fold in folds:
                print(fold)
            print("------------")

            chat_model: BaseChatModel = experiments.chat_model_for_name(model_name)
            print(f"Using model: {chat_model.name}")

            experiments.experiment(
                importer=importer,
                formatters=formatters,
                model_name=model_name,
                chat_model=chat_model,
                storage=storage,
                num_shots=num_shots,
                dry_run=False,
                folds=folds,
                on_new_document=power_logger.set_current_document,
                on_new_fold=power_logger.set_current_fold_id
            )
        else:
            print("Error: incorrect experiment type (use pet_md or pet_re)")

    # parse answers and power logs of all iterations from one experiment and write them into a pandas dataframe
    def parse_results(directory_path: str):
        answer_directory = os.fsencode(directory_path + "answers")
        power_directory = os.fsencode(directory_path + "power")
        answer_data = []
        pow_data = []
        # parsing answers
        for answer_file in os.listdir(answer_directory):
            filepath = directory_path + "answers/" + os.fsdecode(answer_file)
            start_idx = filepath.find("iteration")
            end_idx = filepath.find(".json")
            current_iter = filepath[start_idx + len("iteration"):end_idx]
            experiment_results = experiments.parse.parse_file(filepath)
            num_parse_errors, experiment_stats = parse_experiments(experiment_results,
                                                                   data.PetImporter("res/data/pet/all"
                                                                                    ".new.jsonl"),
                                                                   None, False)
            scores_by_step = get_scores(experiment_stats, False, None)

            for step, printable_scores in scores_by_step.items():
                for tag, score in printable_scores.scores_by_tag.items():
                    answer_data.append({
                        "Tag": tag,
                        "P": score.p,
                        "R": score.r,
                        "F1": score.f1,
                        "iteration": current_iter
                    })
                micro_scores = printable_scores.micro_averaged_scores
                answer_data.append({
                    "Tag": "Micro_Avg",
                    "P": micro_scores.p if micro_scores.p is not None else 0.0,
                    "R": micro_scores.r if micro_scores.r is not None else 0.0,
                    "F1": micro_scores.f1 if micro_scores.f1 is not None else 0.0,
                })

                macro_scores = printable_scores.macro_averaged_scores
                answer_data.append({
                    "Tag": "Macro_Avg",
                    "P": macro_scores.p if macro_scores.p is not None else 0.0,
                    "R": macro_scores.r if macro_scores.r is not None else 0.0,
                    "F1": macro_scores.f1 if macro_scores.f1 is not None else 0.0,
                })


            # parsing power logs
            for power_file in os.listdir(power_directory):
                filepath = directory_path + "power/" + os.fsdecode(power_file)
                start_idx = filepath.find("iteration")
                end_idx = filepath.find(".dat")
                current_iter = filepath[start_idx + len("iteration"):end_idx]
                timestamps, power_measurements = experiments.discrete_integral.parse_logfile(filepath)
                kwh = experiments.discrete_integral.calc_integral_trapezoid(timestamps, power_measurements)
                time_elapsed = round(timestamps[-1] / 1000000)
                average_power = round(statistics.fmean(power_measurements), 6)
                pow_data.append(
                    {
                        "kWh": kwh,
                        "runtime": time_elapsed,
                        "avg. power draw": average_power,
                        "iteration": current_iter
                    }
                )

        ans_df = pd.DataFrame(answer_data)
        ans_df.set_index("Tag", inplace=True)
        pow_df = pd.DataFrame(pow_data)
        return ans_df, pow_df

    # calculate average, standard deviation from given dataframes
    def calc_statistics_from_dataframe(directory_path: str, ans_df=None, pow_df=None):
        complete_df = pd.DataFrame()
        if ans_df is not None:
            pd.concat([complete_df, ans_df])
        if pow_df is not None:
            pd.concat([complete_df, pow_df])

        # add new columns


    exp_type = "pet_md"
    mod_name = "ollama-lamarck-14B"
    total_iter = 2
    date_formatted = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if exp_type == "pet_md":
        dir_path = f"res/efficiency/pet-md/{mod_name}/{date_formatted}/"
    elif exp_type == "pet_re":
        dir_path = f"res/efficiency/pet-re/{mod_name}/{date_formatted}/"
    else:
        dir_path = None
    # dir_path = "/home/fpoeschl/llm-process-generation/res/efficiency/pet-md/ollama-lamarck-14B/2025-02-03_09-04-14/"

    for i in range(1, total_iter + 1):
        log_path = dir_path + f"power/iteration{i}.dat"
        power_logger = power.PowerLogger(run_experiments, log_path, [exp_type, dir_path, mod_name, i])
        power_logger.start_logging()
    answer_df, power_df = parse_results(dir_path)
    pd.set_option('display.max_rows', None)  # Show all rows
    pd.set_option('display.max_columns', None)  # Show all columns
    print(answer_df)
    print(power_df)
