import datetime
import typing

import nltk

import data
import experiments
import format
from experiments import sampling

if __name__ == "__main__":

    def select_one_by_constraint_types(documents: typing.List[data.VanDerAaDocument],
                                       constraint_types: typing.List[str]) -> \
            typing.List[data.VanDerAaDocument]:

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


    def filter_by_constraint_types(documents: typing.List[data.VanDerAaDocument],
                                   constraint_types: typing.List[str]) -> \
            typing.List[data.VanDerAaDocument]:

        if constraint_types is None or len(constraint_types) == 0:
            return documents

        filter_result = []
        for doc in documents:
            if any(c_type for c_type in [c.type for c in doc.constraints] if c_type in constraint_types):
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

        num_shots = 3
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
            #     prompt_path="van-der-aa/re/iterative/minimalistic/succession.txt",
            #     context_tags=[],
            #     only_tags=['succession']
            # ),
            # format.IterativeVanDerAaRelationListingFormattingStrategy(
            #     steps=["constraints"],
            #     separate_tasks=False,
            #     prompt_path="van-der-aa/re/iterative/minimalistic/precedence.txt",
            #     context_tags=[],
            #     only_tags=['precedence']
            # ),
            # format.IterativeVanDerAaRelationListingFormattingStrategy(
            #     steps=["constraints"],
            #     separate_tasks=False,
            #     prompt_path="van-der-aa/re/iterative/minimalistic/response.txt",
            #     context_tags=[],
            #     only_tags=['response']
            # ),
            # format.IterativeVanDerAaRelationListingFormattingStrategy(
            #     steps=["constraints"],
            #     separate_tasks=False,
            #     prompt_path="van-der-aa/re/iterative/minimalistic/init_end.txt",
            #     context_tags=[],
            #     only_tags=['init', 'end']
            # )
            format.IterativeVanDerAaRelationListingFormattingStrategy(
                steps=["constraints"],
                separate_tasks=False,
                prompt_path="van-der-aa/re/minimalistic_explained.txt",
                context_tags=[],
                only_tags=['response', 'init', 'end', 'precedence', 'succession']
            )
        ]

        # # 2 STEP APPROACH
        # formatters = [
        #     format.IterativeVanDerAaRelationListingFormattingStrategy(
        #         steps=["constraints"],
        #         separate_tasks=False,
        #         prompt_path="van-der-aa/re/iterative/2step/init_end_2step.txt",
        #         context_tags=[],
        #         only_tags=['init', 'end']
        #     ),
        #     format.IterativeVanDerAaRelationListingFormattingStrategy(
        #         steps=["constraints"],
        #         separate_tasks=False,
        #         prompt_path="van-der-aa/re/iterative/2step/prec_resp_succ_2step.txt",
        #         context_tags=[],
        #         only_tags=['precedence', 'response', 'succession']
        #     )
        # ]

        importer = data.VanDerAaImporter("res/data/van-der-aa/")

        documents = importer.do_import()
        documents = select_one_by_constraint_types(documents, [])
        # documents = filter_by_constraint_types(documents, ['precedence'])
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
