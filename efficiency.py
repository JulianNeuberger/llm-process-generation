import datetime
import os
import re
import statistics
import pandas as pd
import nltk
from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from pathlib import Path
import data
import experiments
import format
from experiments import sampling, power
from experiments.parse import parse_experiments, get_scores
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
if __name__ == "__main__":
    def run_experiments(experiment_type: str, directory_path: str, model_name: str, iteration: int):

        if experiment_type == "pet_md":
            # copy of pet_md
            load_dotenv()

            # Load sentence tokenizer if necessary
            try:
                nltk.data.find("tokenizers/punkt")
            except LookupError:
                nltk.download("punkt")

            num_shots = 3

            storage = directory_path + f"answers/iteration{iteration}.json"

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

            # formatters = [
            #     format.PetMentionListingFormattingStrategy(
            #         steps=["mentions"],
            #         only_tags=None,
            #         generate_descriptions=False,
            #         prompt="pet/md/unified.txt",
            #     )
            # ]

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

            storage = directory_path + f"answers/iteration{iteration}.json"

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

            formatters = [format.PetRelationListingFormattingStrategy(steps=["relations"])]
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
    def parse_results(directory_path: str, parse_tags: bool):
        answer_directory = os.fsencode(directory_path + "answers")
        power_directory = os.fsencode(directory_path + "power")
        answer_data = []
        pow_data = []
        total_parse_errors = []
        # parsing answers
        for answer_file in os.listdir(answer_directory):
            filepath = directory_path + "answers/" + os.fsdecode(answer_file)
            start_idx = filepath.find("iteration")
            end_idx = filepath.find(".json")
            current_iter = int(filepath[start_idx + len("iteration"):end_idx])
            experiment_results = experiments.parse.parse_file(filepath)
            parse_errors, experiment_stats = parse_experiments(experiment_results,
                                                               data.PetImporter("res/data/pet/all"
                                                                                ".new.jsonl"),
                                                               None, False)
            total_parse_errors.append(parse_errors)
            scores_by_step = get_scores(experiment_stats, False, None)

            for step, printable_scores in scores_by_step.items():
                if parse_tags:
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
                    "iteration": current_iter
                })

                macro_scores = printable_scores.macro_averaged_scores
                answer_data.append({
                    "Tag": "Macro_Avg",
                    "P": macro_scores.p if macro_scores.p is not None else 0.0,
                    "R": macro_scores.r if macro_scores.r is not None else 0.0,
                    "F1": macro_scores.f1 if macro_scores.f1 is not None else 0.0,
                    "iteration": current_iter
                })

            # parsing power logs
        for power_file in os.listdir(power_directory):
            filepath = directory_path + "power/" + os.fsdecode(power_file)
            start_idx = filepath.find("iteration")
            end_idx = filepath.find(".dat")
            current_iter = int(filepath[start_idx + len("iteration"):end_idx])
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
        return ans_df, pow_df, total_parse_errors


    def boxplot_from_data(directory_path: str, ans_df, pow_df, parse_err, retries):
        parts = directory_path.split("ollama-")[1].split("/")
        model_name = parts[0]
        # use micro averaged scores for plotting
        ans_df_micro = ans_df.loc[ans_df.index == 'Micro_Avg']
        ans_df_melted = ans_df_micro.reset_index().melt(id_vars=['Tag', 'iteration'], value_vars=['P', 'R', 'F1'],
                                                        var_name='Metric', value_name='Value')
        plt.figure(figsize=(8, 6))
        plt.title(f"{model_name}")
        sns.boxplot(x='Metric', y='Value', data=ans_df_melted, width=0.5, showmeans=True)
        save_path_ans = directory_path + f"ans-boxplot-{model_name}.png"
        plt.savefig(save_path_ans, dpi=300, bbox_inches='tight')
        plt.close()

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

        sns.boxplot(data=parse_err, width=0.5, showmeans=True, ax=axes[0])
        axes[0].set_ylabel('Total parse errors')

        sns.boxplot(data=retries, width=0.5, showmeans=True, ax=axes[1])
        axes[1].set_ylabel('Total number of retries')

        plt.title(f"{model_name}")
        plt.tight_layout()
        save_path_err = directory_path + f"err-boxplot-{model_name}.png"
        plt.savefig(save_path_err, dpi=300, bbox_inches='tight')
        plt.close()

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # 1 row, 3 columns

        # Plot kWh
        sns.boxplot(data=pow_df['kWh'], width=0.5, showmeans=True, ax=axes[0])
        axes[0].set_ylabel('Energy usage in kWh')

        # Plot runtime
        sns.boxplot(data=pow_df['runtime'], width=0.5, showmeans=True, ax=axes[1])
        axes[1].set_ylabel('Runtime in s')

        # Plot avg. power draw
        sns.boxplot(data=pow_df['avg. power draw'], width=0.5, showmeans=True, ax=axes[2])
        axes[2].set_ylabel('Avg. Power Draw in W')
        plt.title(f"{model_name}")
        plt.tight_layout()

        save_path_pow = directory_path + f"pow-boxplot-{model_name}.png"
        plt.savefig(save_path_pow, dpi=300, bbox_inches='tight')
        plt.close()

    # parse answer files for total number of retries and return them as a list
    def parse_retries(directory_path: str):
        answer_directory = os.path.join(directory_path, "answers")  # FIXED: No os.fsencode()

        retries = []
        # parsing answers
        for answer_file in os.listdir(answer_directory):
            answer_file = answer_file.decode("utf-8") if isinstance(answer_file, bytes) else answer_file
            filepath = os.path.join(answer_directory, answer_file)

            with open(filepath, 'r', encoding='utf-8') as file:
                file_content = file.read()  # Read the content of the file

                # Use regular expressions to find all occurrences of 'num_tries' in the file
                num_tries_matches = re.findall(r'"num_tries":\s*\[([^]]+)]', file_content)
                total_sum = 0
                for match in num_tries_matches:
                    # Convert the matched string into a list of integers
                    num_tries = [int(x) for x in match.split(",")]
                    filtered_values = [num - 1 for num in num_tries if isinstance(num, int) and num > 1]
                    total_sum += sum(filtered_values)
            retries.append(total_sum)
        return retries

    # helper function for calculating average and standard deviation and returning them as a tuple
    def mean_std(series):
        mean = round(series.mean(), 4)
        std = round(series.std(), 4)
        return mean, std

    # plot power draw over time and mark when a new document is loaded
    # set short_flag to only print the first 5% of data (approx. 3 documents)
    def plot_power_draw(directory_path: str, percentage: float):
        power_directory = os.fsencode(directory_path + "power")
        parts = directory_path.split("ollama-")[1].split("/")
        model_name = parts[0]
        for power_file in os.listdir(power_directory):
            filepath = directory_path + "power/" + os.fsdecode(power_file)
            start_idx = filepath.find("iteration")
            end_idx = filepath.find(".dat")
            current_iter = filepath[start_idx + len("iteration"):end_idx]
            data = []

            with open(filepath, 'r') as file:
                lines = file.readlines()
                num_elements = len(lines)
                lines = lines[:int(num_elements * percentage)]
                for line in lines:
                    parts = re.split(r'\t+', line.strip())
                    timestamp = datetime.strptime(parts[0], "%Y-%m-%d %H:%M:%S.%f")
                    document = parts[1]
                    power_draw = float(parts[3])
                    data.append((timestamp, document, power_draw))

            df = pd.DataFrame(data, columns=["Timestamp", "Document", "Value"])

            # Convert the timestamps to seconds
            df["Time_Seconds"] = (df["Timestamp"] - df["Timestamp"].iloc[0]).dt.total_seconds()

            plt.figure(figsize=(10, 5))
            plt.plot(df["Time_Seconds"], df["Value"], linestyle='-', label="Power Draw")

            prev_doc = None
            first_change = True  # don't plot a line at the start
            for j, row in df.iterrows():
                if row["Document"] and row["Document"] != prev_doc:
                    if not first_change:
                        plt.axvline(row["Time_Seconds"], color='r', linestyle='--')
                    prev_doc = row["Document"]
                    first_change = False
            average_power = df["Value"].mean()
            plt.axhline(average_power, color='b', linestyle='--', label=f"Avg Power: {average_power:.2f} W")
            plt.xlabel("Time [s}")
            plt.ylabel("Power Draw [W]")
            plt.title(f"{model_name}")
            plt.legend()
            plt.xticks(rotation=45)

            save_path = directory_path + f"power_draw-plot-{model_name}-iter" + current_iter + ".png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        return

    # apply mean_std to dataframes and save as Excel file
    def calc_statistics_from_dataframe(directory_path: str, ans_df, pow_df, parse_errs: list, retries: list):
        parts = directory_path.split("ollama-")[1].split("/")
        model_name = parts[0]
        # Combine results of all iterations for power df
        combined_pow = pow_df.agg(
            {
                'kWh': mean_std,
                'runtime': mean_std,
                'avg. power draw': mean_std
            }
        )
        values = combined_pow.values.flatten()
        new_pow_df = pd.DataFrame([values],
                                  columns=["kWh (mean, std)", "runtime (mean, std)", "average power draw (mean, std)"])
        ans_df_reset = ans_df.reset_index()

        # Combine results of all iterations for answer df
        combined_ans = ans_df_reset.groupby('Tag').agg({
            'P': mean_std,
            'R': mean_std,
            'F1': mean_std,
            'iteration': 'max'
        })
        combined_ans.columns = ['P (mean, std)', 'R (mean, std)', 'F1 (mean, std)', 'iterations']

        combined_ans.reset_index(inplace=True)

        # Combine parse errors and retries
        parse_series = pd.Series(parse_errs)
        retry_series = pd.Series(retries)

        parse_stats = mean_std(parse_series)
        retry_stats = mean_std(retry_series)
        err_data = {
            'Parse errors': [parse_stats],
            'Number of retries': [retry_stats]
        }
        err_df = pd.DataFrame(err_data)

        with pd.ExcelWriter(directory_path + f"stats_calculated-{model_name}.xlsx", engine='xlsxwriter') as writer:
            # Save the DataFrames to separate sheets
            new_pow_df.to_excel(writer, sheet_name='Power_Stats', index=False)
            combined_ans.to_excel(writer, sheet_name='Answer_Stats', index=False)
            err_df.to_excel(writer, sheet_name='Error_Stats', index=False)

    # write data from single iterations to an excel-file for T-Test
    def to_excel(directory_path, ans_df, pow_df, parse_errs: list, retries: list):
        parts = directory_path.split("ollama-")[1].split("/")
        model_name = parts[0]
        file_path = directory_path + f"stats_by_iteration-{model_name}.xlsx"
        # use micro averaged scores
        ans_df_micro = ans_df.loc[ans_df.index == 'Micro_Avg']
        # convert lists to df
        err_df_combined = pd.DataFrame({
            'parse_err': parse_errs,
            'retries': retries
        })
        with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
            ans_df_micro.to_excel(writer, sheet_name="answers", index=False)
            pow_df.to_excel(writer, sheet_name="power", index=False)
            err_df_combined.to_excel(writer, sheet_name="errors", index=False)


    exp_type = "pet_md"
    mod_name = "ollama-mistral-large-123b-instruct-2407-q2_K"
    total_iter = 5
    date_formatted = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # dir_path direkt angeben (Ordner der "answers" und "power" beinhaltet)
    # dir_path = "res/efficiency/pet-md/ollama-ultiima-72b-Q4_K_S/2025-02-14_21-51-44"
    # if not dir_path.endswith("/"):
    #     dir_path += "/"

    # diesen Teil auskommentieren und dir_path oben direkt angeben, falls nur Statistiken und Plots benötigt werden
    if exp_type == "pet_md":
        dir_path = f"res/efficiency/pet-md/{mod_name}/{date_formatted}/"
    elif exp_type == "pet_re":
        dir_path = f"res/efficiency/pet-re/{mod_name}/{date_formatted}/"
    else:
        dir_path = None

    for i in range(1, total_iter + 1):
        log_path = dir_path + f"power/iteration{i}.dat"
        log_file = Path(log_path)
        if log_file.is_file():
            break
        power_logger = power.PowerLogger(run_experiments, log_path, [exp_type, dir_path, mod_name, i])
        power_logger.start_logging()

    # unbenötigte Statistiken auskommentieren
    #     answer_df, power_df, list_parse_errors = parse_results(dir_path, False)
    #     list_retries = parse_retries(dir_path)
    #     to_excel(dir_path, answer_df, power_df, list_parse_errors, list_retries)
    #     boxplot_from_data(dir_path, answer_df, power_df, list_parse_errors, list_retries)
    #     calc_statistics_from_dataframe(dir_path, answer_df, power_df, list_parse_errors, list_retries)
    #     plot_power_draw(dir_path, 0.05)
