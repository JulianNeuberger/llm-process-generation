import json
import os
import data
import random
from annotate_md import annotate_md
from annotate_re import annotate_re
from annotate_er import annotate_er
from pathlib import Path
from create_descriptions import create_topics, create_bp_from_llm, save_to_folder, create_bp_list_from_llm
import experiments
from dotenv import load_dotenv
import nltk
from langchain_core.language_models import BaseChatModel
from data.pet import PetJsonExporter

run_number = "jerex_02"
base_path = f"res/data/annotate/pipe/run_{run_number}/"
text_path = base_path + "0_text"
json_path = base_path + "1_base"
annotate_md_path = base_path + "2_annotate_md"
annotate_md_extract_path = base_path + "3_annotate_md_extract"
annotate_md_wo_doubles_path = base_path + "4_annotate_md_wo_doubles"
annotate_re_path = base_path + "5_annotate_re"
annotate_re_extract_path = base_path + "6_annotate_re_extract"
annotate_er_path = base_path + "7_annotate_er"
annotate_er_extract_path = base_path + "8_annotate_er_extract"


def main():
    #test_double()
    fullPipe()
    #print_doubles()
    #combine_1_file("complete.json")
    #remove_newline("res/data/annotate/pipe/run_10/2_annotated/Board_Meeting_Preparation_and_Execution.json")

def fullPipe():
    load_dotenv()
    # Load sentence tokenizer if necessary
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    # model_name = "local_llm/llama3.1:70b"

    model_name = "gpt-4o"
    and_prompt_r = "rand"
    examples_prompt_b = True
    count = 10

    gpt_chat_model: BaseChatModel = experiments.chat_model_for_name(model_name, 0)

    # setting up folder structure
    #create_dirs()

    # Create descriptions and save topics
    if False:
        tlist = create_topics(count, gpt_chat_model)
        with open(base_path + "topics.json", "w") as file:
            json.dump(tlist, file)
        hlist = create_bp_list_from_llm(tlist,gpt_chat_model)
        with open(base_path + "headings.json", "w") as file:
            json.dump(hlist, file)
        print("Anzahl themen", len(hlist))
        text_list = create_bp_from_llm(and_prompt_r, examples_prompt_b, count, gpt_chat_model, "", hlist)
        save_to_folder(base_path + "0_text", text_list)

    # Convert descriptions to json
    if False:
        dir_list = os.listdir(text_path)
        for dir_name in dir_list:
            txt_dir = text_path + "/" + dir_name
            json_dir = base_path + "1_base/" + dir_name[:-4] + ".json"
            import_txt = data.AnnotateImporter(txt_dir)
            pet_json_exporter = PetJsonExporter(json_dir)
            pet_json_exporter.export(import_txt.get_pedDoc())

    # annotate_md and extract
    if False:
        dir_list = os.listdir(json_path)
        for json_data in dir_list:
            storage = annotate_md_path + "/" + json_data
            experiment_list, importer = annotate_md(gpt_chat_model, json_path + "/" + json_data, storage)
            pred_doc, steps = experiments.get_predicted_doc(storage, importer)
            json_data = json_data.replace(" ", "_")
            data.PetJsonExporter(annotate_md_extract_path + "/" + json_data).export([pred_doc])

    # remove double tokens
    if True:
        dir_list = os.listdir(annotate_md_extract_path)
        for md_data in dir_list:
            doc = data.PetImporter(annotate_md_extract_path + "/" + md_data).do_import()
            double_dict = experiments.get_double_assigned_tokens(doc[0])
            print(len(double_dict), "doubled mention indices deleted")
            experiments.double_assigned_remove(double_dict, doc[0])
            data.PetJsonExporter(annotate_md_wo_doubles_path + "/" + md_data).export(doc)

    # annotate_re and extract
    if True:
        dir_list = os.listdir(annotate_md_wo_doubles_path)
        for er_data in dir_list:
            storage = annotate_re_path + "/" + er_data
            experiment_list, importer = annotate_re(gpt_chat_model, annotate_md_wo_doubles_path + "/" + er_data,
                                                    storage)
            pred_doc, steps = experiments.get_predicted_doc(storage, importer)
            data.PetJsonExporter(annotate_re_extract_path + "/" + er_data).export([pred_doc])

    # annotate_er and extract
    if True:
        dir_list = os.listdir(annotate_re_extract_path)
        for md_data in dir_list:
            storage = annotate_er_path + "/" + md_data
            experiment_list, importer = annotate_er(gpt_chat_model, annotate_re_extract_path + "/" + md_data, storage)
            pred_doc, steps = experiments.get_predicted_doc(storage, importer)
            data.PetJsonExporter(annotate_er_extract_path + "/" + md_data).export([pred_doc])
            combine_1_file("complete.jsonl")




def er_step():
    model_name = "gpt-4o"
    run_number = 1
    gpt_chat_model: BaseChatModel = experiments.chat_model_for_name(model_name, 1)
    create_dirs()
    dir_list = os.listdir(annotate_md_extract_path)
    for md_data in dir_list:
        storage = annotate_er_path + "/" + md_data
        experiment_list, importer = annotate_er(gpt_chat_model, annotate_md_extract_path + "/" + md_data, storage)
        pred_doc, steps = experiments.get_predicted_doc(storage, importer)
        data.PetJsonExporter(annotate_er_extract_path + "/" + md_data).export([pred_doc])


def create_dirs():
    Path(json_path).mkdir(parents=True, exist_ok=True)
    Path(annotate_md_path).mkdir(parents=True, exist_ok=True)
    Path(annotate_md_extract_path).mkdir(parents=True, exist_ok=True)
    Path(annotate_md_wo_doubles_path).mkdir(parents=True, exist_ok=True)
    Path(annotate_er_path).mkdir(parents=True, exist_ok=True)
    Path(annotate_er_extract_path).mkdir(parents=True, exist_ok=True)
    Path(annotate_re_path).mkdir(parents=True, exist_ok=True)
    Path(annotate_re_extract_path).mkdir(parents=True, exist_ok=True)
    print("Dir structure created")


def test_double():
    model_name = "gpt-4o"
    gpt_chat_model: BaseChatModel = experiments.chat_model_for_name(model_name, 1)
    annotate_md_path_test = base_path + "8_annotate_md_test"
    Path(annotate_md_path_test).mkdir(parents=True, exist_ok=True)
    annotate_md_extract_path_test = base_path + "9_annotate_md_extract_test"
    Path(annotate_md_extract_path_test).mkdir(parents=True, exist_ok=True)

    dir_list = os.listdir(json_path)
    for json_data in dir_list:
        storage = annotate_md_path_test + "/" + json_data
        experiment_list, importer = annotate_md(gpt_chat_model, json_path + "/" + json_data, storage)
        pred_doc, steps = experiments.get_predicted_doc(storage, importer)
        data.PetJsonExporter(annotate_md_extract_path_test + "/" + json_data).export([pred_doc])

    # print double tokens
    dir_list = os.listdir(annotate_md_extract_path_test)
    for md_data in dir_list:
        doc = data.PetImporter(annotate_md_extract_path_test + "/" + md_data).do_import()
        double_dict = experiments.get_double_assigned_tokens(doc[0])
        for k, v in double_dict.items():
            print("index: ", k, " | type: ", v)

def remove_doubles():
    dir_list = os.listdir(annotate_md_extract_path)
    for md_data in dir_list:
        print(md_data)
        doc = data.PetImporter(annotate_md_extract_path + "/" + md_data).do_import()
        double_dict = experiments.get_double_assigned_tokens(doc[0])
        for k, v in double_dict.items():
            print("index: ", k, " | type: ", v)
        experiments.double_assigned_remove(double_dict, doc[0])
        print(md_data)
        double_dict = experiments.get_double_assigned_tokens(doc[0])
        if not double_dict:
            print("no doubles")
        else:
            for k, v in double_dict.items():
                print("index: ", k, " | type: ", v)
        data.PetJsonExporter(annotate_md_extract_path + "/" + md_data).export(doc)

def print_doubles():
    dir_list = os.listdir(annotate_md_extract_path)
    for md_data in dir_list:
        print(md_data)
        doc = data.PetImporter(annotate_md_extract_path + "/" + md_data).do_import()
        double_dict = experiments.get_double_assigned_tokens(doc[0])
        if not double_dict:
            print("no doubles")
        else:
            for k, v in double_dict.items():
                print("index: ", k, " | type: ", v)


def combine_1_file( file_name: str, dir_path = annotate_er_extract_path):
    dir_list = os.listdir(dir_path)
    docs = [None]* len(dir_list)
    for x in range(len(dir_list)):
        doc = data.PetImporter(dir_path + "/" + dir_list[x]).do_import()
        for docu in doc:
            docs[x] = docu
    data.PetJsonExporter(base_path + file_name).export(docs)

def remove_newline(file_name: str):
    with open(file_name, 'r') as file:
        data = file.read()
    print(data)
    data = data.replace("\n", "")
    print(data)

main()