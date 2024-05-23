import datetime

import langchain_openai
import nltk
from langchain_core.language_models import BaseChatModel

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
        storage = f"res/answers/van-der-aa-md/{date_formatted}.json"
        # storage = f"res/answers/pet/2024-02-27_13-29-40.json"

        num_shots = 0
        model_name = "gpt-4-0125-preview"

        # formatter = format.PetMentionListingFormattingStrategy(["mentions"])
        # formatter = format.PetTagFormattingStrategy()
        formatter = format.VanDerAaMentionListingFormattingStrategy(
            steps=["mentions"],
            prompt="van-der-aa/md/default.txt",
        )
        importer = data.VanDerAaImporter("res/data/van-der-aa/")

        documents = importer.do_import()
        print(f"Dataset consists of {len(documents)} documents.")
        folds = sampling.generate_folds(documents, num_shots)

        print("Using folds:")
        print("------------")
        for fold in folds:
            print(fold)
        print("------------")

        chat_model: BaseChatModel = langchain_openai.ChatOpenAI(
            model_name=model_name, temperature=0
        )

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
