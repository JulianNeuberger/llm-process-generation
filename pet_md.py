import datetime

import nltk

import data
import experiments
import format
from experiments import sampling

if __name__ == "__main__":

    def main():
        # Load sentence tokenizer if necessary
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")

        date_formatted = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        storage = f"res/answers/pet-md/{date_formatted}.json"
        # storage = f"res/answers/pet-md/2024-03-14_18-42-49.json"

        num_shots = 1
        model_name = "gpt-4-0125-preview"

        # formatter = format.PetMentionListingFormattingStrategy(["mentions"])
        importer = data.PetImporter("res/data/pet/all.new.jsonl")
        train_docs = [d.id for d in importer.do_import() if d.id != "doc-6.1"]
        # folds = [{"train": train_docs, "test": ["doc-6.1"]}]
        folds = sampling.generate_folds(importer.do_import(), num_shots)

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
                prompt="pet/md/iterative/with_explanation/activity.txt",
            ),
            format.IterativePetMentionListingFormattingStrategy(
                ["mentions"],
                "actor",
                context_tags=["activity"],
                prompt="pet/md/iterative/with_explanation/actor.txt",
            ),
            format.IterativePetMentionListingFormattingStrategy(
                ["mentions"],
                "activity data",
                context_tags=["activity", "actor"],
                prompt="pet/md/iterative/with_explanation/activity_data.txt",
            ),
            format.IterativePetMentionListingFormattingStrategy(
                ["mentions"],
                "further specification",
                context_tags=["activity", "actor", "activity data"],
                prompt="pet/md/iterative/with_explanation/further_specification.txt",
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
                prompt="pet/md/iterative/with_explanation/xor_gateway.txt",
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
                prompt="pet/md/iterative/with_explanation/condition_specification.txt",
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
                prompt="pet/md/iterative/with_explanation/and_gateway.txt",
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

        experiments.experiment(
            importer=importer,
            formatters=formatters,
            model_name=model_name,
            storage=storage,
            num_shots=num_shots,
            dry_run=False,
            folds=folds,
        )

        experiments.print_experiment_results(storage, importer, verbose=True)

    main()
