import datetime
import typing

import nltk

import data
import experiments
import format
from experiments import sampling

if __name__ == "__main__":

    def filter_by_constraint_types(documents: typing.List[data.VanDerAaDocument], constraint_types: typing.List[str]) -> \
            typing.List[data.VanDerAaDocument]:
        if constraint_types is None or len(constraint_types) == 0:
            return documents
        filter_result = []
        for doc in documents:
            if any(x in [c.type for c in doc.constraints] for x in constraint_types):
                filter_result.append(doc)
        return filter_result


    def main():
        # Load sentence tokenizer if necessary
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")

        date_formatted = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        storage = f"res/answers/van-der-aa-re/{date_formatted}.json"
        # storage = f"res/answers/pet/2024-02-27_13-29-40.json"

        num_shots = 0
        model_name = "gpt-4-0125-preview"

        # formatter = format.PetMentionListingFormattingStrategy(["mentions"])
        # formatter = format.PetTagFormattingStrategy()
        # formatter = format.VanDerAaRelationListingFormattingStrategy(
        #     steps=["constraints"],
        #     separate_tasks=False,
        #     prompt_path="van-der-aa/re/step-wise_tuning_CCoT.txt",
        # )
        formatters = [
            # format.IterativeVanDerAaRelationListingFormattingStrategy(
            #     steps=["constraints"],
            #     separate_tasks=False,
            #     prompt_path="van-der-aa/re/iterative/precedence.txt",
            #     context_tags=[],
            #     only_tags=['precedence']
            # ),
            # format.IterativeVanDerAaRelationListingFormattingStrategy(
            #     steps=["constraints"],
            #     separate_tasks=False,
            #     prompt_path="van-der-aa/re/iterative/response.txt",
            #     context_tags=['precedence'],
            #     only_tags=['response']
            # ),
            format.IterativeVanDerAaRelationListingFormattingStrategy(
                steps=["constraints"],
                separate_tasks=False,
                prompt_path="van-der-aa/re/iterative/init_end.txt",
                context_tags=[],
                only_tags=['init', 'end']
            ),
            format.IterativeVanDerAaRelationListingFormattingStrategy(
                steps=["constraints"],
                separate_tasks=False,
                prompt_path="van-der-aa/re/iterative/succession",
                context_tags=[],
                only_tags=['succession']
            )
        ]

        importer = data.VanDerAaImporter("res/data/van-der-aa/")

        documents = importer.do_import()
        documents = filter_by_constraint_types(documents, ['end', 'init'])[4:11]
        print(f"Dataset consists of {len(documents)} documents.")
        folds = sampling.generate_folds(documents, num_shots)

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
