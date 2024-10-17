import os

from data.pet import PetToken

import nltk.tag
from nltk import tokenize

CUR_DIR = os.path.dirname(os.path.realpath(__file__))
PROMPT_DIR = os.path.normpath(os.path.join(CUR_DIR, "..", "res", "prompts"))
DATA_DIR = os.path.normpath(os.path.join(CUR_DIR, "..", "res", "data", "annotate"))


def load_prompt_from_file(file_path: str) -> str:
    file_path = os.path.join(PROMPT_DIR, file_path)
    with open(file_path, "r") as f:
        return f.read()
def create_PetToken_from_txt(file_path: str) -> PetToken:
    with open(file_path, 'r') as file:
        data = file.read().replace('\n', ' ')
    petList = []
    word_counter = 0
    sentence_counter = 0
    sentences = tokenize.sent_tokenize(data)
    for sentence in sentences:
        words = tokenize.word_tokenize(sentence)
        pos_tags = nltk.tag.pos_tag(words)
        for x in range(len(words)):
            petList.append(PetToken(words[x], word_counter, pos_tags[x][1], sentence_counter))
            word_counter += 1
        sentence_counter += 1
    return petList


