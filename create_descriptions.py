import typing
import langchain_openai
import random

import nltk
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pathlib import Path
from langchain_core.language_models import BaseChatModel
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.output_parsers import PydanticOutputParser
import experiments
import datetime
from format.common import tokensize_from_text
from format.prompts.bp_prompt import base_prompt, and_prompt, examples_prompt, topics_prompt, bp_list_prompt


class BP_Description(BaseModel):
    heading: str = Field(description="A heading for the business process description.")
    text: str = Field(description="The text of the business process description. The snippet shouldn't contain the headline.")


class BP_Topics(BaseModel):
    topic: str = Field(description="The topic of a business")


class BP_List(BaseModel):
    topics: typing.List[BP_Topics] = Field (description="List of topics")


class BP_heading(BaseModel):
    heading: str = Field(description="The heading of a process")


class BP_headings_List(BaseModel):
    headings: typing.List[BP_heading] = Field(description="List of headings")


def main():
    load_dotenv()
    # Load sentence tokenizer if necessary
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    #model_name = "local_llm/llama3.1:70b"

    model_name = "gpt-4o"
    and_prompt_b = True
    examples_prompt_b = True
    count = 10
    chat_model: BaseChatModel = experiments.chat_model_for_name(model_name,1)

    tlist = create_topics(count, chat_model)
    heading_list = create_bp_list_from_llm(tlist, chat_model)
    for heading in heading_list:
        print(heading)

    print(len(heading_list))
    #text_list = create_bp_from_llm(and_prompt_b, examples_prompt_b, count, chat_model,"",heading_list)

    #save_to_folder("res/data/annotate/pipe/0_txt/test", text_list)

def create_bp_list_from_llm(topic_list: typing.List, chat_model)-> typing.List:
    pydantic_parser = PydanticOutputParser(pydantic_object=BP_headings_List)
    format_instructions = pydantic_parser.get_format_instructions()
    prompt_template = PromptTemplate(
        input_variables=["topic"],
        template=bp_list_prompt,
        partial_variables={"format_instructions": format_instructions}
    )
    heading_list = []
    for topic in topic_list:
        prompt = prompt_template.format_prompt(
            topic=topic
        )
        res = chat_model.invoke(prompt.to_string())
        parsed_output = pydantic_parser.parse(res.content)
        for heading in parsed_output.headings:
            heading_list.append(heading.heading)
    return heading_list

def create_bp_from_llm(and_prompt_r, examples_prompt_b: bool, count: int, chat_model: BaseChatModel, storage: str, topiclist):
    text_list = []
    counter = 0
    pydantic_parser = PydanticOutputParser(pydantic_object=BP_Description)
    format_instructions = pydantic_parser.get_format_instructions()
    prompt_template =PromptTemplate(
        input_variables=["and_prompt", "examples_prompt","topic"],
        template=base_prompt,
        partial_variables={"format_instructions": format_instructions}
    )
    for topic in topiclist:
        if and_prompt_r == "rand":
            if (random.random() < 0.5):
                and_prompt_b = True
            else:
                and_prompt_b = False
        elif and_prompt_r:
            and_prompt_b = True
        elif not and_prompt_r:
            and_prompt_b = False
        prompt = prompt_template.format_prompt(
            and_prompt=and_prompt if and_prompt_b else "",
            examples_prompt=examples_prompt if examples_prompt_b else "",
            topic=topic
        )
        res = chat_model.invoke(prompt.to_string())
        parsed_output = pydantic_parser.parse(res.content)
        text_list.append(parsed_output)
        print("Creating text", counter, "/", len(topiclist))
        counter += 1
    return text_list

def save_to_folder(folder_path: str, text_List: typing.List[BP_Description]):
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    for text in text_List:
        print("heading: "+ text.heading)
        print("text: " + text.text)
        print("Token size:")
        print( tokensize_from_text(text.text))
        with open(folder_path + "/" + text.heading + ".txt","w") as text_file:
            text_file.write(text.text)

def create_topics(count: int, chat_model: BaseChatModel):
    pydantic_parser = PydanticOutputParser(pydantic_object=BP_List)
    format_instructions = pydantic_parser.get_format_instructions()
    prompt_template = PromptTemplate(
        input_variables=["count"],
        template=topics_prompt,
        partial_variables={"format_instructions": format_instructions}
    )
    prompt = prompt_template.format_prompt(count=count)

    res = chat_model.invoke(prompt.to_string())

    parsed_output = pydantic_parser.parse(res.content)
    topic_list = []
    for topic in parsed_output.topics:
        topic_list.append(topic.topic)
    return topic_list

