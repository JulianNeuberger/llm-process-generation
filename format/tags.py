import re
import typing

import data
import format


prompt = """You are a business process modelling expert, tasked with identifying mentions of
process relevant elements in textual descriptions of business processes. These mentions are 
spans of text, that are of a certain type, as described below: 

- Actor: a person, department, or similar role that participates actively in the business 
         process, e.g., "the student", "the professor", "the judge", "the clerk". It should
         only be extracted, if the current sentence describes the actor executing a task. 
         Include the determiner if it is present, e.g. extract "the student" from "First the 
         student studies for their exam". Can also be a pronoun, such as "he", "I", "she".
- Activity: an active task or action executed during the business process, e.g., 
            "examine", "write", "bake", "review". Do not extract, if it is information about
            the process itself, such as "the process ends", as this is not a task in the process!
            Can also be an event during the process, that is executed by an external, implicit actor, 
            that is not mentioned in the text, e.g., "placed" in "when an order is placed".
            Never contains the Actor that executes it, nor the Activity Data that's used during this
            Activity! 
- Activity Data: a physical object, or digital data that is relevant to the process, because 
                 an action produces or uses it, e.g., "the paper", "the report", "the machine part".
                 Always include the determiner! Can also be a pronoun, like "it", but is always part of 
                 a task description. Is never information about the process itself, as in "the process 
                 ends", here "process" is not Activity Data!
- Further Specification: important information about an activity, such as the mean, the manner 
                         of execution, or how an activity is executed. Follows an Activity in the
                         same sentence, and describes how an Activity (task) is being done.
- XOR Gateway: textual representation of a decision in the process, usually triggered by adverbs or 
               conjunctions, e.g., "if", "otherwise", "when"
- AND Gateway: textual description of parallel work streams in the process, i.e., simultaneous 
               actions performed, marked by phrases like, e.g., "while", "meanwhile", "at the same time" 
- Condition Specification: defines the condition of an XOR Gateway path usually directly following the
                           gateway trigger word, e.g., "it is ready", "the claim is valid",
                           "temperature is above 180"

You can retrieve mentions by surrounding them with xml-style tags
e.g. "<activity> format </activity> <activity_data> the text </activity_data> ."
Use underscores in mention types instead of spaces, e.g. 'Activity_Data' instead of 'Activity Data'.
Spans may not be nested! Forbidden: "<activity> format <activity_data> the text </activity_data> </activity> ."

Put out raw text, without code formatting.

Always leave spaces around each word, tag, and punctuation, under no circumstances 
alter the original text besides inserting the tags."""


generated_prompt = """
Given a textual description of a business process, identify and annotate the following elements: 
actors (who performs actions), activities (what actions are performed), 
activity data (specific data or objects involved in the activities), 
further specifications (additional details about activities or conditions), 
XOR gateways (points in the process where decisions lead to diverging paths based on conditions), 
and condition specifications (the conditions leading to those paths). 
Use the tags <Actor>, <Activity>, <Activity_Data>, <Further_Specification>, <XOR_Gateway>, and 
<Condition_Specification> to mark each element respectively. 
Ensure to insert tags into the original text, do not change it in any other way. Do not remove spaces, 
or fix typing errors. Insert spaces around tags. Accurately highlight these elements according to their 
definitions."""


class PetTagFormattingStrategy(format.BaseFormattingStrategy[data.PetDocument]):
    """
    Format for mention detection using LLMs, where mentions and their position
    are marked by XML-style opening and closing tags,
    e.g. "<activity> format </activity> the <business_object> text </business_object>"
    """

    def __init__(
        self,
        steps: None = None,
        only_tags: typing.List[str] = None,
        include_ids: bool = False,
    ):
        super().__init__(["mentions"])
        self._include_ids = include_ids
        self._only_tags = only_tags
        if self._only_tags is not None:
            self._only_tags = [t.lower() for t in self._only_tags]

    @staticmethod
    def supported_steps() -> typing.List[typing.Literal["mentions"]]:
        return ["mentions"]

    def description(self) -> str:
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
            if self._only_tags is not None and mention.type not in self._only_tags:
                continue
            attributes = {}
            if self._include_ids:
                attributes["id"] = str(i)
            opening_tag, closing_tag = self.ner_to_tag(mention.type, attributes)
            token_texts.insert(mention.token_document_indices[-1] + 1, closing_tag)
            token_texts.insert(mention.token_document_indices[0], opening_tag)

        return " ".join(token_texts)

    def parse(self, document: data.PetDocument, string: str) -> data.PetDocument:
        document = document.copy(clear=["mentions", "entities", "relations"])

        tokens = string.split(" ")
        opening_tag_regex = re.compile(r"<([^/>]+)>")
        closing_tag_regex = re.compile(r"</([^>]+)>")
        reversed_tokens = list(reversed(tokens))
        for token_index, token in enumerate(reversed_tokens):
            token_index = len(reversed_tokens) - token_index - 1
            if self.split_off_tag(token, token_index, tokens, closing_tag_regex):
                continue
            if self.split_off_tag(token, token_index, tokens, opening_tag_regex):
                continue

        tokens = [t for t in tokens if t != ""]
        mentions: typing.List[data.PetMention] = []
        current_mention_indices: typing.Optional[typing.List[int]] = None
        current_mention_type: typing.Optional[str] = None
        index_in_document = 0
        for token in tokens:
            if opening_tag_regex.match(token):
                # start new mention
                ner_tag = self.tag_to_ner(token)
                assert (
                    current_mention_type is None
                ), f"Unclosed mention with type {current_mention_type} in doc {document.id} at '{token}'"
                current_mention_type = ner_tag
                current_mention_indices = []
                continue

            if closing_tag_regex.match(token):
                # finish mentions
                assert current_mention_type is not None
                assert current_mention_indices is not None
                assert len(current_mention_indices) > 0
                mentions.append(
                    data.PetMention(
                        type=current_mention_type.lower().strip(),
                        token_document_indices=tuple(current_mention_indices),
                    )
                )
                current_mention_type = None
                current_mention_indices = None
                continue

            original_token_text = document.tokens[index_in_document].text
            assert (
                original_token_text == token
            ), f"Read token '{token}', but expected '{original_token_text}', the LLM seems to have altered the text."
            if current_mention_indices is not None:
                current_mention_indices.append(index_in_document)
            index_in_document += 1

        document.mentions = mentions
        return document

    @staticmethod
    def split_off_tag(
        token: str, token_index: int, tokens: typing.List[str], tag_re: re.Pattern
    ) -> bool:
        match = re.search(tag_re, token)
        if match is None:
            return False
        before = token[: match.start(0)]
        tag = token[match.start(0) : match.end(0)]
        after = token[match.end(0) :]

        if before == "" and after == "":
            return False

        if after != "":
            tokens.insert(token_index + 1, after)
        tokens[token_index] = tag
        if before != "":
            tokens.insert(token_index, before)

        return True

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
        documents = data.PetImporter("../res/data/pet/all.new.jsonl").do_import()
        formatter = PetTagFormattingStrategy()
        for d in documents:
            print("Input:")
            print(formatter.input(d))
            print()
            print("Output:")
            print(formatter.output(d))
            print()
            print("-------")
            print()

    main()
