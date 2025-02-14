import datetime

import langchain_anthropic
import langchain_openai
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

        # model_name = "gpt-4-0125-preview"
        model_name = "gpt-4o-2024-05-13"
        # model_name = "claude-3-sonnet-20240229"
        # model_name = "claude-3-opus-20240229"

        date_formatted = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        storage = f"res/answers/{model_name}/van-der-aa-md/{date_formatted}.json"
        # storage = (
        #     "res/answers/claude-3-opus-20240229/van-der-aa-md/2024-05-23_15-23-35.json"
        # )

        # formatter = format.PetMentionListingFormattingStrategy(["mentions"])
        # formatter = format.PetTagFormattingStrategy()
        formatter = format.VanDerAaMentionListingFormattingStrategy(
            steps=["mentions"],
            prompt="van-der-aa/md/default.txt",
        )
        importer = data.VanDerAaImporter("res/data/van-der-aa/")

        documents = importer.do_import()
        print(f"Dataset consists of {len(documents)} documents.")
        folds = sampling.generate_folds(
            documents, num_shots, strategy="similarity", seed=42
        )

        print("Using folds:")
        print("------------")
        for fold in folds:
            print(fold)
        print("------------")

        chat_model: BaseChatModel = experiments.chat_model_for_name(model_name)

        experiments.experiment(
            importer=importer,
            formatters=[formatter],
            model_name=model_name,
            chat_model=chat_model,
            storage=storage,
            num_shots=num_shots,
            dry_run=False,
            folds=folds,
        )

        experiments.print_experiment_results(storage, importer, verbose=True)

    main()
