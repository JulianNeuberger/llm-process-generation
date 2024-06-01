import datetime

import nltk
from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel

import data
import experiments
import format
from experiments import sampling

if __name__ == "__main__":

    def main():
        load_dotenv()

        # Load sentence tokenizer if necessary
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")

        num_shots = 3

        # model_name = "gpt-4-turbo-2024-04-09"
        # model_name = "gpt-4o-2024-05-13"
        # model_name = "claude-3-sonnet-20240229"
        # model_name = "claude-3-opus-20240229"
        # model_name = "meta-llama/Meta-Llama-3-70B-Instruct"
        # model_name = "deepinfra/airoboros-70b"
        # model_name = "gpt-4-0125-preview"
        # model_name = "Qwen/Qwen1.5-72B-Chat"
        model_name = "gpt-3.5-turbo-0125"
        # model_name = "mistral-large-latest"

        date_formatted = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        storage = f"res/answers/{model_name}/pet-md/{date_formatted}.json"
        # storage = "res/answers/claude-3-opus-20240229/pet-md/2024-05-28_14-46-19.json"
        # storage = "res/answers/gpt-4-0125-preview/pet-md/2024-05-23_13-31-08.json"

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
        )

        experiments.print_experiment_results(storage, importer, verbose=True)

    main()
