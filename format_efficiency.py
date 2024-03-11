import typing

import langchain_openai
import matplotlib.pyplot as plt
import seaborn as sns

import data
import format

documents = data.PetImporter("res/data/pet/all.new.jsonl").do_import()
document = [d for d in documents if d.id == "doc-6.1"][0]

model_name = "gpt-4-0125-preview"
chat_model: langchain_openai.ChatOpenAI = langchain_openai.ChatOpenAI(
    model_name=model_name, temperature=0
)

num_tokens_by_formatter: typing.Dict[str, int] = {}

formatters: typing.Dict[str, format.BaseFormattingStrategy] = {
    "TSV": format.PetMentionListingFormattingStrategy(["mentions"]),
    "tags": format.PetTagFormattingStrategy(),
    "efficient YAML": format.PetEfficientYamlFormattingStrategy(["mentions"]),
    "YAML": format.PetYamlFormattingStrategy(["mentions"]),
    "JSON": format.PetJsonifyFormattingStrategy(["mentions"]),
}
for name, formatter in formatters.items():
    formatted = formatter.output(document.copy([]))
    num_tokens = chat_model.get_num_tokens(formatted)
    num_tokens_by_formatter[name] = num_tokens

max_len_name = max(len(k) for k in num_tokens_by_formatter)
print(
    "\n".join(f"{k:<{max_len_name}} : {v}" for k, v in num_tokens_by_formatter.items())
)

sns.set_theme()
sns.barplot(
    x=list(num_tokens_by_formatter.keys()), y=list(num_tokens_by_formatter.values())
)
plt.ylabel("number of output tokens")
plt.xlabel("formatting strategy")
plt.savefig("figures/formats/token-efficiency.pdf")
plt.savefig("figures/formats/token-efficiency.png")
