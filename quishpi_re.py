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
        # storage = f"res/answers/quishpi-re/{date_formatted}.json"
        storage = f"res/answers/quishpi-re/2024-03-14_16-45-25.json"

        num_shots = 3
        model_name = "gpt-4-0125-preview"

        # formatter = format.PetMentionListingFormattingStrategy(["mentions"])
        # formatter = format.PetTagFormattingStrategy()
        formatter = format.VanDerAaRelationListingFormattingStrategy(
            steps=["constraints"],
            separate_tasks=False,
            prompt_path="quishpi/re/step-wise-detailed.txt",
        )
        # importer = data.VanDerAaImporter("res/data/quishpi/csv")
        importer = data.VanDerAaSentenceImporter("res/data/quishpi/csv")
        documents = importer.do_import()
        # documents = [d for d in documents if d.id in ['4-1_intaker_workflow-1-20', '4-1_intaker_workflow-1-23', '2-1_sla_violation-1-22', '4-1_intaker_workflow-1-10', '6-1_acme-1-4', 'datacollection_3-1-16', '4-1_intaker_workflow-1-4', '4-1_intaker_workflow-1-9', 'datacollection_1-1-10', '3-2_2009-2_conduct_directions_hearing-1-0', 'datacollection_3-1-6', 'datacollection_1-1-26', '1-1_bicycle_manufacturing-1-7', 'datacollection_2-1-6', '2-1_sla_violation-1-35', 'datacollection_3-1-19', '10-2_process_b3-1-8', '6-1_acme-1-2', '2-1_sla_violation-1-1', '1364308140_rev4-1-3', 'datacollection_3-1-29', '3-6_2010-1_claims_notification-1-0', 'datacollection_3-1-15', '1120589054_rev4-1-5', 'datacollection_1-1-37', '10-2_process_b3-1-13', 'datacollection_1-1-30', 'datacollection_3-1-48', '20818304_rev1-1-1', '7-1_calling_leads-1-2', 'datacollection_3-1-32', '1120589054_rev4-1-1', 'datacollection_2-1-10', 'datacollection_1-1-36', 'datacollection_1-1-14', 'datacollection_3-1-9', '5-1_active_vos_tutorial-1-3', '4-1_intaker_workflow-1-21', 'datacollection_1-1-31', '2-1_sla_violation-1-23', '1-1_bicycle_manufacturing-1-6', '6-1_acme-1-9', 'datacollection_3-1-28', '1-2_computer_repair-1-5', 'datacollection_1-1-1', '3-6_2010-1_claims_notification-1-4', '10-2_process_b3-1-12', 'datacollection_1-1-35', 'datacollection_1-1-29', 'datacollection_3-1-46', '784358570_rev2-1-7', '784358570_rev2-1-8', 'datacollection_3-1-13', 'datacollection_1-1-18', '1081511532_rev3-1-2', '1-1_bicycle_manufacturing-1-11', 'datacollection_3-1-3', 'datacollection_3-1-27', 'datacollection_2-1-15', '3-2_2009-2_conduct_directions_hearing-1-1', '7-1_calling_leads-1-5', '1120589054_rev4-1-0', '4-1_intaker_workflow-1-13', '1-1_bicycle_manufacturing-1-5', 'datacollection_2-1-0', '2-1_sla_violation-1-36', 'datacollection_1-1-27', '2-1_sla_violation-1-15', '1364308140_rev4-1-1', '10-2_process_b3-1-6', '7-1_calling_leads-1-6', 'datacollection_3-1-37', '1-1_bicycle_manufacturing-1-9', 'datacollection_3-1-24', '4-1_intaker_workflow-1-37', '6-1_acme-1-16', '9-2_exercise_2-1-4', '2-1_sla_violation-1-16', 'datacollection_3-1-18', '784358570_rev2-1-2', '4-1_intaker_workflow-1-22', '1-1_bicycle_manufacturing-1-1', '3-6_2010-1_claims_notification-1-2', 'datacollection_3-1-40', '10-2_process_b3-1-1', 'datacollection_1-1-19', '4-1_intaker_workflow-1-2', '5-1_active_vos_tutorial-1-2', 'datacollection_3-1-14', '7-1_calling_leads-1-3', '1081511532_rev3-1-0', '4-1_intaker_workflow-1-11', 'datacollection_3-1-26', 'datacollection_1-1-0', 'datacollection_3-1-44', '4-1_intaker_workflow-1-33', '4-1_intaker_workflow-1-12', '3-6_2010-1_claims_notification-1-3', '2-1_sla_violation-1-5', '6-1_acme-1-0']]
        # [0:98]
        # 0:6 = 0:98
        # 6:15
        print(f"Dataset consists of {len(documents)} documents.")
        folds = sampling.generate_sentence_constraint_folds(documents, num_shots)
        # folds = [f for f in folds if f['test'][0] in ['4-1_intaker_workflow-1-20', '4-1_intaker_workflow-1-23', '2-1_sla_violation-1-22', '4-1_intaker_workflow-1-10', '6-1_acme-1-4', 'datacollection_3-1-16', '4-1_intaker_workflow-1-4', '4-1_intaker_workflow-1-9', 'datacollection_1-1-10', '3-2_2009-2_conduct_directions_hearing-1-0', 'datacollection_3-1-6', 'datacollection_1-1-26', '1-1_bicycle_manufacturing-1-7', 'datacollection_2-1-6', '2-1_sla_violation-1-35', 'datacollection_3-1-19', '10-2_process_b3-1-8', '6-1_acme-1-2', '2-1_sla_violation-1-1', '1364308140_rev4-1-3', 'datacollection_3-1-29', '3-6_2010-1_claims_notification-1-0', 'datacollection_3-1-15', '1120589054_rev4-1-5', 'datacollection_1-1-37', '10-2_process_b3-1-13', 'datacollection_1-1-30', 'datacollection_3-1-48', '20818304_rev1-1-1', '7-1_calling_leads-1-2', 'datacollection_3-1-32', '1120589054_rev4-1-1', 'datacollection_2-1-10', 'datacollection_1-1-36', 'datacollection_1-1-14', 'datacollection_3-1-9', '5-1_active_vos_tutorial-1-3', '4-1_intaker_workflow-1-21', 'datacollection_1-1-31', '2-1_sla_violation-1-23', '1-1_bicycle_manufacturing-1-6', '6-1_acme-1-9', 'datacollection_3-1-28', '1-2_computer_repair-1-5', 'datacollection_1-1-1', '3-6_2010-1_claims_notification-1-4', '10-2_process_b3-1-12', 'datacollection_1-1-35', 'datacollection_1-1-29', 'datacollection_3-1-46', '784358570_rev2-1-7', '784358570_rev2-1-8', 'datacollection_3-1-13', 'datacollection_1-1-18', '1081511532_rev3-1-2', '1-1_bicycle_manufacturing-1-11', 'datacollection_3-1-3', 'datacollection_3-1-27', 'datacollection_2-1-15', '3-2_2009-2_conduct_directions_hearing-1-1', '7-1_calling_leads-1-5', '1120589054_rev4-1-0', '4-1_intaker_workflow-1-13', '1-1_bicycle_manufacturing-1-5', 'datacollection_2-1-0', '2-1_sla_violation-1-36', 'datacollection_1-1-27', '2-1_sla_violation-1-15', '1364308140_rev4-1-1', '10-2_process_b3-1-6', '7-1_calling_leads-1-6', 'datacollection_3-1-37', '1-1_bicycle_manufacturing-1-9', 'datacollection_3-1-24', '4-1_intaker_workflow-1-37', '6-1_acme-1-16', '9-2_exercise_2-1-4', '2-1_sla_violation-1-16', 'datacollection_3-1-18', '784358570_rev2-1-2', '4-1_intaker_workflow-1-22', '1-1_bicycle_manufacturing-1-1', '3-6_2010-1_claims_notification-1-2', 'datacollection_3-1-40', '10-2_process_b3-1-1', 'datacollection_1-1-19', '4-1_intaker_workflow-1-2', '5-1_active_vos_tutorial-1-2', 'datacollection_3-1-14', '7-1_calling_leads-1-3', '1081511532_rev3-1-0', '4-1_intaker_workflow-1-11', 'datacollection_3-1-26', 'datacollection_1-1-0', 'datacollection_3-1-44', '4-1_intaker_workflow-1-33', '4-1_intaker_workflow-1-12', '3-6_2010-1_claims_notification-1-3', '2-1_sla_violation-1-5', '6-1_acme-1-0']]

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
