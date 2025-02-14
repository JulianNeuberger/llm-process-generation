import os

CUR_DIR = os.path.dirname(os.path.realpath(__file__))
PROMPT_DIR = os.path.normpath(os.path.join(CUR_DIR, "..", "res", "prompts"))


def load_prompt_from_file(file_path: str) -> str:
    file_path = os.path.join(PROMPT_DIR, file_path)
    with open(file_path, "r") as f:
        return f.read()
