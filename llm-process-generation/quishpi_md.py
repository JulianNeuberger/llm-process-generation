import datetime

import nltk
from langchain_core.language_models import BaseChatModel

from dotenv import load_dotenv

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

        num_shots = 0

        # model_name = "gpt-4-0125-preview"
        model_name = "gpt-4o-2024-05-13"
        # model_name = "claude-3-sonnet-20240229"
        # model_name = "claude-3-opus-20240229"

        date_formatted = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        storage = f"res/answers/{model_name}/quishpi-md/{date_formatted}.json"
        # storage = (
        #     f"res/answers/claude-3-opus-20240229/quishpi-md/2024-05-23_15-28-59.json"
        # )

        importer = data.QuishpiImporter("res/data/quishpi", exclude_tags=["entity"])
        # folds = [{"train": [], "test": ["20818304_rev1"]}]
        folds = sampling.generate_folds(
            importer.do_import(), num_shots, strategy="similarity", seed=42
        )

        # formatters = [
        #     format.QuishpiMentionListingFormattingStrategy(
        #         ["mentions"], prompt="quishpi/md/long-no-explain.txt"
        #     )
        # ]

        formatters = [
            format.IterativeQuishpiMentionListingFormattingStrategy(
                ["mentions"], tag="condition", context_tags=[]
            ),
            format.IterativeQuishpiMentionListingFormattingStrategy(
                ["mentions"], tag="action", context_tags=["condition"]
            ),
        ]

        print("Using folds:")
        print("------------")
        for fold in folds:
            print(fold)
        print("------------")

        # chat_model: BaseChatModel = langchain_openai.ChatOpenAI(
        #     model_name=model_name, temperature=0
        # )

        chat_model: BaseChatModel = experiments.chat_model_for_name(model_name)

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
