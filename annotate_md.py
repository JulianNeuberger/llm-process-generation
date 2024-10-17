import datetime

import nltk
from langchain_core.language_models import BaseChatModel

from dotenv import load_dotenv
import os
import data
import experiments
import format
import json
from data.pet import PetDictExporter
from experiments import sampling




if __name__ == "__main__":

    def main():

        # Load sentence tokenizer if necessary
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")

        load_dotenv()
        model_name = "gpt-4o-mini"
        num_shots = 0
        date_formatted = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        storage = f"res/answers/{model_name}/annotate/{date_formatted}.json"
        file_name = "Inquiry_Offer_Order"
        file_path = file_name + ".txt"
        print(file_path)
        import_txt = data.AnnotateImporter(file_path)
        petDict_exporter = PetDictExporter()
        json_object = petDict_exporter.export_document(import_txt.get_pedDoc()[0])
        print(json_object)
        file_json = f"res/data/annotate/{file_name}.jsonl"
        with open(file_json, "w") as outfile:
            json.dump(json_object,outfile)
        importer = data.PetImporter(file_json)

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
        )

        #experiments.print_experiment_results(storage, importer, verbose=True)

    main()