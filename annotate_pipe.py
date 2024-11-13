import annotate_re
import annotate_md
from create_descriptions import create_topics, create_bp_from_llm
from langchain_core.language_models import BaseChatModel
import experiments
from dotenv import load_dotenv
import nltk


def main():
    fullPipe()

def fullPipe():
    model_Name = "gpt-4o"
    load_dotenv()
    # Load sentence tokenizer if necessary
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    # model_name = "local_llm/llama3.1:70b"

    model_name = "gpt-4o"
    and_prompt_b = True
    examples_prompt_b = True
    count = 2
    chat_model: BaseChatModel = experiments.chat_model_for_name(model_name, 1)

    create_topics(count, chat_model)

    # text_list = create_bp_from_llm(and_prompt_b, examples_prompt_b, count, chat_model,"")

    # save_to_folder("", text_list)


main()

