import datetime
import typing

import langchain_openai
import nltk
from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel

import data
import experiments
import format
from experiments import sampling

if __name__ == "__main__":

    def select_one_by_constraint_types(
        documents: typing.List[data.VanDerAaDocument],
        constraint_types: typing.List[str],
    ) -> typing.List[data.VanDerAaDocument]:

        if constraint_types is None or len(constraint_types) == 0:
            return documents

        filter_result = []
        for doc in documents:
            constraint_types_in_document = [c.type for c in doc.constraints]
            doc_added = False
            for c_type in constraint_types_in_document:
                if c_type in constraint_types and not doc_added:
                    filter_result.append(doc)
                    doc_added = True
                    constraint_types.remove(c_type)
                elif c_type in constraint_types:
                    constraint_types.remove(c_type)
            if len(constraint_types) == 0:
                return filter_result
        return filter_result

    def filter_by_constraint_types(
        documents: typing.List[data.VanDerAaDocument],
        constraint_types: typing.List[str],
    ) -> typing.List[data.VanDerAaDocument]:

        if constraint_types is None or len(constraint_types) == 0:
            return documents

        filter_result = []
        for doc in documents:
            if any(
                c_type
                for c_type in [c.type for c in doc.constraints]
                if c_type in constraint_types
            ):
                filter_result.append(doc)

        return filter_result

    def main():
        load_dotenv()

        # Load sentence tokenizer if necessary
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")

        num_shots = 3
        model_name = "gpt-4-0125-preview"

        date_formatted = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        storage = f"res/answers/{model_name}/van-der-aa-re/{date_formatted}.json"

        formatters = [
            format.VanDerAaRelationListingFormattingStrategy(
                steps=["constraints"],
                separate_tasks=False,
                prompt_path="van-der-aa/re/step-wise_tuning_CCoT_single.txt",
                context_tags=None,
                only_tags=None,
            )
        ]

        importer = data.VanDerAaSentenceImporter("res/data/van-der-aa/")

        documents = importer.do_import()
        documents = select_one_by_constraint_types(documents, [])
        print(f"Dataset consists of {len(documents)} documents.")
        folds = sampling.generate_sentence_constraint_folds(documents, num_shots)

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
            formatters=formatters,
            model_name=model_name,
            chat_model=chat_model,
            storage=storage,
            dry_run=False,
            folds=folds,
            num_shots=num_shots,
        )

        experiments.print_experiment_results(storage, importer, verbose=True)

    main()
