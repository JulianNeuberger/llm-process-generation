import re
import typing

import data
import format


prompt = """You are a business process modelling expert, tasked with identifying mentions of
process relevant elements in textual descriptions of business processes. These mentions are 
spans of text, that are of a certain type, as described below: 

- Actor: a person, department, or similar role that participates actively in the business 
         process, e.g., "the student", "the professor", "the judge", "the clerk"  
- Activity: a task or action executed by an actor during the business process, e.g., 
            "examine", "write", "bake", "review"
- Activity Data: a physical object, or digital data that is relevant to the process, because 
                 an action produces or uses it, e.g., "the paper", "the report", "the machine part"
- Further Specification: important information about an activity, such as the mean, the manner 
                         of execution, or how an activity is executed.
- XOR Gateway: textual representation of a decision in the process, usually triggered by adverbs or 
               conjunctions, e.g., "if", "otherwise", "when"
- AND Gateway: textual description of parallel work streams in the process, i.e., simultaneous 
               actions performed, marked by phrases like, e.g., "while", "meanwhile", "at the same time" 
- Condition Specification: defines the condition of an XOR-gateway path usually following the
                           gateway trigger word, e.g., "it is ready", "the claim is valid",
                           "temperature is above 180"

You can retrieve mentions by surrounding them with xml-style tags
e.g. "<activity> format </activity> <activity_data> the text </activity_data>.
Use underscores in mention types instead of spaces, e.g. 'Activity_Data' instead of 'Activity Data'.

Put out raw text, without code formatting.

Always leave spaces around each word, tag, and punctuation, under no circumstances 
alter the original text besides inserting text."""


THasMentions


class PetTagFormattingStrategy(format.BaseFormattingStrategy[data.PetDocument]):
    """
    Format for mention detection using LLMs, where mentions and their position
    are marked by XML-style opening and closing tags,
    e.g. "<activity> format </activity> the <business_object> text </business_object>"
    """

    def __init__(self, steps: None = None, include_ids: bool = False):
        super().__init__(["mentions"])
        self._include_ids = include_ids

    @staticmethod
    def supported_steps() -> typing.List[typing.Literal["mentions"]]:
        return ["mentions"]

    @staticmethod
    def description() -> str:
        return prompt

    def input(self, document: data.PetDocument) -> str:
        return " ".join([t.text for t in document.tokens])

    def output(self, document: data.PetDocument) -> str:
        token_texts = [t.text for t in document.tokens]
        mentions: typing.List[typing.Tuple[int, data.PetMention]] = list(
            enumerate(m.copy() for m in document.mentions)
        )
        # sort so that last mentions in text come first (not by id)
        mentions.sort(key=lambda m: -m[1].token_document_indices[0])

        # insert tags, starting from behind, so we dont have to
        # adjust token indices of mentions...
        for i, mention in mentions:
            attributes = {}
            if self._include_ids:
                attributes["id"] = str(i)
            opening_tag, closing_tag = self.ner_to_tag(mention.type, attributes)
            token_texts.insert(mention.token_document_indices[-1] + 1, closing_tag)
            token_texts.insert(mention.token_document_indices[0], opening_tag)

        return " ".join(token_texts)

    def parse(self, document: data.PetDocument, string: str) -> data.PetDocument:
        document = document.copy(
            clear_mentions=True, clear_entities=True, clear_relations=True
        )

        tokens = string.split(" ")
        opening_tag_regex = re.compile(r"<([^/>]+)>")
        closing_tag_regex = re.compile(r"</([^>]+)>")

        mentions: typing.List[data.PetMention] = []
        current_mention: typing.Optional[data.PetMention] = None
        index_in_document = 0
        for token in tokens:
            if opening_tag_regex.match(token):
                # start new mention
                ner_tag = self.tag_to_ner(token)
                assert current_mention is None
                current_mention = data.PetMention(ner_tag=ner_tag)
                continue

            if closing_tag_regex.match(token):
                # finish mentions
                assert current_mention is not None
                assert len(current_mention.token_document_indices) > 0
                mentions.append(current_mention)
                current_mention = None
                continue

            original_token_text = document.tokens[index_in_document].text
            assert (
                original_token_text == token
            ), f"Read token '{token}', but expected '{original_token_text}', the LLM seems to have altered the text."
            if current_mention is not None:
                current_mention.token_document_indices.append(index_in_document)
            index_in_document += 1

        document.mentions = mentions
        return document

    @staticmethod
    def ner_to_tag(
        ner: str, attributes: typing.Dict[str, str]
    ) -> typing.Tuple[str, str]:
        """
        Converts NER tags to xml style opening and closing tags.

        Examples
        ------------------

        > ner_to_tag("activity")
        > <activity>, </activity>

        > ner_to_tag("further specification")
        > <further_specification>, </further_specification>

        :param ner: named entity recognition tag
        :param attributes: dictionary of attributes to include in opening tag

        :return: tuple of opening and closing tags
        """
        ner = ner.replace(" ", "_")
        if len(attributes) > 0:
            formatted_attributes = " ".join([f"{k}={v}" for k, v in attributes.items()])
            return f"<{ner} {formatted_attributes}>", f"</{ner}>"
        return f"<{ner}>", f"</{ner}>"

    @staticmethod
    def tag_to_ner(tag: str) -> str:
        # strip < and >
        tag = tag[1:-1]
        if "=" in tag:
            # contains attributes
            tag, _ = tag.split(" ", 1)
        return tag.replace("_", " ")


if __name__ == "__main__":

    def main():
        documents = data.PetImporter("res/pet/all.new.jsonl").do_import()
        formatter = PetTagFormattingStrategy()
        for d in documents:
            print(formatter.output(d))
            print("-------")
            print()

    main()
