import re
import typing

import data
from format import base, common, tags


class VanDerAaMentionListingFormattingStrategy(
    base.BaseFormattingStrategy[data.VanDerAaDocument]
):
    def __init__(self, steps: typing.List[str], prompt: str = None):
        super().__init__(steps)
        if prompt is None:
            prompt = "van-der-aa/md/default.txt"
        self._prompt = prompt

    @property
    def args(self):
        return {"prompt": self._prompt}

    def description(self) -> str:
        return common.load_prompt_from_file(self._prompt)

    def output(self, document: data.VanDerAaDocument) -> str:
        return "\n".join([m.text for m in document.mentions])

    def input(self, document: data.VanDerAaDocument) -> str:
        return document.text

    def parse(self, document: data.VanDerAaDocument, string: str) -> base.ParseResult:
        document = document.copy(clear=["mentions"])
        for mention_text in string.splitlines(keepends=False):
            mention_text = mention_text.strip()
            if mention_text == "":
                continue
            document.mentions.append(data.VanDerAaMention(text=mention_text))
        return base.ParseResult(document, 0)


class VanDerAaRelationListingFormattingStrategy(
    base.BaseFormattingStrategy[data.VanDerAaDocument]
):
    def __init__(
        self,
        steps: typing.List[typing.Literal["constraints"]],
        prompt_path: str,
        separate_tasks: bool,
        context_tags: typing.List[str] = None,
        only_tags: typing.List[str] = None,
    ):
        super().__init__(steps)
        self._only_tags = only_tags
        self._context_tags = context_tags
        self._prompt_path = prompt_path
        self._prompt = common.load_prompt_from_file(prompt_path)
        self._separate_tasks = separate_tasks
        self._sentence_re = re.compile(
            r"^\s*\*\*\s?sentence\s*(\d+)\s*\*\*\s*$", flags=re.IGNORECASE
        )

    def description(self) -> str:
        return self._prompt

    @property
    def args(self):
        return {
            "prompt_path": self._prompt_path,
            "separate_tasks": self._separate_tasks,
            "context_tags": self._context_tags,
            "only_tags": self._only_tags,
        }

    def _dump_constraints(
        self, constraints: typing.List[data.VanDerAaConstraint], sentence_id: int
    ) -> str:
        res = []
        for c in constraints:
            if c.sentence_id != sentence_id:
                continue
            if self._only_tags is not None and c.type.lower() not in self._only_tags:
                continue
            negative = "TRUE" if c.negative else "FALSE"
            tail = ""
            if c.tail is not None:
                tail = c.tail.text
            if self._separate_tasks:
                res.append(f"{negative}\t{c.type}\t{c.head.text}\t{tail}")
            else:
                res.append(
                    f"{c.sentence_id}\t{negative}\t{c.type}\t{c.head.text}\t{tail}"
                )
        return "\n".join(res)

    @staticmethod
    def _dump_actions(document: data.VanDerAaDocument, sentence_id: int) -> str:
        actions: typing.Set[str] = set()
        for c in document.constraints:
            if c.sentence_id != sentence_id:
                continue
            actions.add(c.head.text)
            if c.tail is not None:
                actions.add(c.tail.text)
        return "\n".join(actions)

    def output(self, document: data.VanDerAaDocument) -> str:
        res = []
        for i, sentence in enumerate(document.sentences):
            if self._separate_tasks:
                res.append(f"** Sentence {i} **")
                res.append("")
                res.append("Actions:")
                res.append(self._dump_actions(document, i))
                res.append("")
                res.append("Constraints:")
            res.append(self._dump_constraints(document.constraints, i))
            if self._separate_tasks:
                res.append("")
        return "\n".join(res)

    def input(self, document: data.VanDerAaDocument) -> str:
        sentences = "\n".join(
            f"Sentence {i}: {s}" for i, s in enumerate(document.sentences)
        )

        if self._context_tags is None:
            return sentences

        relevant_constraints = []
        for c in document.constraints:
            if c.type.lower() not in self._context_tags:
                continue
            relevant_constraints.append(c)
        constraints = []
        for i in range(len(document.sentences)):
            constraints.append(self._dump_constraints(relevant_constraints, i))
        constraints = "\n".join(constraints)
        return f"{sentences}\n\n{constraints}"

    def parse(self, document: data.VanDerAaDocument, string: str) -> base.ParseResult:
        constraints = []
        current_sentence_id: typing.Optional[int] = None
        num_errors = 0
        for line in string.splitlines(keepends=False):
            line = line.strip()
            if line == "":
                continue

            match = re.match(self._sentence_re, line)
            if match is not None:
                # new sentence
                current_sentence_id = int(match.group(1))
                continue

            if "\t" not in line:
                # either a header like "Actions:" or "Constraints:",
                # or an action, which we currently do not parse
                continue

            split_line = line.strip().split("\t")
            if self._separate_tasks:
                if len(split_line) == 4:
                    negative, c_type, c_head, c_tail = split_line
                    c_head = data.VanDerAaMention(text=c_head)
                    c_tail = data.VanDerAaMention(text=c_tail)
                elif len(split_line) == 3:
                    negative, c_type, c_head = split_line
                    c_head = data.VanDerAaMention(text=c_head)
                    c_tail = None
                else:
                    print(
                        f'Expected 3 or 4 tab separated values in line "{line}", got {len(split_line)}, skipping line.'
                    )
                    continue
            else:
                print(split_line)
                if len(split_line) == 5:
                    current_sentence_id, negative, c_type, c_head, c_tail = split_line
                    c_head = data.VanDerAaMention(text=c_head)
                    if c_tail == "":
                        c_tail = None
                    else:
                        c_tail = data.VanDerAaMention(text=c_tail)
                elif len(split_line) == 4:
                    current_sentence_id, negative, c_type, c_head = split_line
                    c_head = data.VanDerAaMention(text=c_head)
                    c_tail = None
                else:
                    print(
                        f'Expected 4 or 5 tab separated values in line "{line}", got {len(split_line)}, skipping line.'
                    )
                    continue
                try:
                    current_sentence_id = int(current_sentence_id)
                except ValueError:
                    num_errors += 1
                    continue

            if c_type.strip() == "":
                print(f"Predicted empty type in {line}. Skipping.")
                continue

            if c_tail == "None":
                raise AssertionError()

            constraints.append(
                data.VanDerAaConstraint(
                    sentence_id=current_sentence_id,
                    type=c_type.strip().lower(),
                    head=c_head,
                    tail=c_tail,
                    negative=negative.lower() == "true",
                )
            )
        doc = data.VanDerAaDocument(
            id=document.id,
            name=document.name,
            text=document.text,
            constraints=constraints,
            sentences=document.sentences,
            mentions=document.mentions,
        )
        return base.ParseResult(doc, num_errors)


class QuishpiREListingFormattingStrategy(VanDerAaRelationListingFormattingStrategy):
    def __init__(self, steps):
        super().__init__(
            steps, prompt_path="quishpi/re/standardized.txt", separate_tasks=True
        )


class IterativeVanDerAaRelationListingFormattingStrategy(
    base.BaseFormattingStrategy[data.VanDerAaDocument]
):
    def __init__(
        self,
        steps: typing.List[typing.Literal["constraints"]],
        prompt_path: str,
        separate_tasks: bool,
        context_tags: typing.List[str] = None,
        only_tags: typing.List[str] = None,
    ):
        super().__init__(steps)
        self._only_tags = only_tags
        self._context_tags = context_tags
        self._prompt_path = prompt_path
        self._prompt = common.load_prompt_from_file(prompt_path)
        self._separate_tasks = separate_tasks
        self._sentence_re = re.compile(
            r"^\s*\*\*\s?sentence\s*(\d+)\s*\*\*\s*$", flags=re.IGNORECASE
        )

    def description(self) -> str:
        return self._prompt

    @property
    def args(self):
        return {
            "prompt_path": self._prompt_path,
            "separate_tasks": self._separate_tasks,
            "context_tags": self._context_tags,
            "only_tags": self._only_tags,
        }

    def _dump_constraints(
        self, constraints: typing.List[data.VanDerAaConstraint]
    ) -> str:
        res = []
        for c in constraints:
            # if c.sentence_id != sentence_id:
            #     continue
            negative = "TRUE" if c.negative else "FALSE"
            tail = ""
            if c.tail is not None:
                tail = c.tail.text
            if self._separate_tasks:
                res.append(f"{negative}\t{c.type}\t{c.head}\t{tail}")
            else:
                res.append(
                    f"{c.sentence_id}\t{negative}\t{c.type}\t{c.head.text}\t{tail}"
                )
        return "\n".join(res)

    @staticmethod
    def _filter_constraints(
        tags_of_interest: typing.List[str],
        constraints: typing.List[data.VanDerAaConstraint],
    ) -> typing.List[data.VanDerAaConstraint]:
        relevant_constraints = []
        for c in constraints:
            if tags_of_interest is None:
                continue
            if c.type.lower() not in tags_of_interest:
                continue
            relevant_constraints.append(c)
        return relevant_constraints

    @staticmethod
    def _dump_actions(document: data.VanDerAaDocument, sentence_id: int) -> str:
        actions: typing.Set[str] = set()
        for c in document.constraints:
            if c.sentence_id != sentence_id:
                continue
            actions.add(c.head.text)
            if c.tail is not None:
                actions.add(c.tail.text)
        return "\n".join(actions)

    def output(self, document: data.VanDerAaDocument) -> str:
        res = []
        relevant_constraints = (
            IterativeVanDerAaRelationListingFormattingStrategy._filter_constraints(
                tags_of_interest=self._only_tags, constraints=document.constraints
            )
        )
        for i, sentence in enumerate(document.sentences):
            if self._separate_tasks:
                res.append(f"Sentence {i}: {sentence}")
        if relevant_constraints is not None and len(relevant_constraints) > 0:
            res.append("Constraints:")
            res.append(self._dump_constraints(relevant_constraints))
        if self._separate_tasks:
            res.append("")
        return "\n".join(res)

    def input(self, document: data.VanDerAaDocument) -> str:
        if len(document.sentences) > 1:
            sentences = "\n".join(
                f"Sentence {i}: {s}" for i, s in enumerate(document.sentences)
            )
        else:
            sentences = document.sentences[0]
        relevant_constraints = (
            IterativeVanDerAaRelationListingFormattingStrategy._filter_constraints(
                tags_of_interest=self._context_tags, constraints=document.constraints
            )
        )
        constraints = [self._dump_constraints(relevant_constraints)]
        # for i in range(len(document.sentences)):
        constraints_as_string = "\n".join(constraints)
        if len(relevant_constraints) > 0:
            constraints_heading = "\nConstraints:\n"
        else:
            constraints_heading = ""
        return f"{sentences}\n{constraints_heading}{constraints_as_string}"

    def parse(self, document: data.VanDerAaDocument, string: str) -> base.ParseResult:
        constraints = []
        current_sentence_id: typing.Optional[int] = None
        for line in string.splitlines(keepends=False):
            if line.strip() == "":
                continue

            match = re.match(self._sentence_re, line)
            if match is not None:
                # new sentence
                current_sentence_id = int(match.group(1))
                continue

            if "\t" not in line:
                # either a header like "Actions:" or "Constraints:",
                # or an action, which we currently do not parse
                continue

            split_line = line.strip().split("\t")
            if self._separate_tasks:
                if len(split_line) == 4:
                    negative, c_type, c_head, c_tail = split_line
                    c_head = data.VanDerAaMention(text=c_head)
                    c_tail = data.VanDerAaMention(text=c_tail)
                elif len(split_line) == 3:
                    negative, c_type, c_head = split_line
                    c_head = data.VanDerAaMention(text=c_head)
                    c_tail = None
                else:
                    print(
                        f'Expected 3 or 4 tab separated values in line "{line}", got {len(split_line)}, skipping line.'
                    )
                    continue
            else:
                if len(split_line) == 5:
                    current_sentence_id, negative, c_type, c_head, c_tail = split_line
                    c_head = data.VanDerAaMention(text=c_head)
                    if c_tail == "":
                        c_tail = None
                    else:
                        c_tail = data.VanDerAaMention(text=c_tail)
                elif len(split_line) == 4:
                    current_sentence_id, negative, c_type, c_head = split_line
                    c_head = data.VanDerAaMention(text=c_head)
                    c_tail = None
                else:
                    print(
                        f'Expected 4 or 5 tab separated values in line "{line}", got {len(split_line)}, skipping line.'
                    )
                    continue
                current_sentence_id = int(current_sentence_id)

            if c_type.strip() == "":
                print(f"Predicted empty type in {line}. Skipping.")
                continue

            if c_tail == "None":
                raise AssertionError()

            constraints.append(
                data.VanDerAaConstraint(
                    sentence_id=current_sentence_id,
                    type=c_type.strip().lower(),
                    head=c_head,
                    tail=c_tail,
                    negative=negative.lower() == "true",
                )
            )
        doc = data.VanDerAaDocument(
            id=document.id,
            name=document.name,
            text=document.text,
            constraints=constraints,
            sentences=document.sentences,
            mentions=document.mentions,
        )
        return base.ParseResult(doc, 0)


class IterativeVanDerAaSelectiveRelationExtractionRefinementStrategy(
    base.BaseFormattingStrategy[data.VanDerAaDocument]
):
    def __init__(
        self,
        steps: typing.List[typing.Literal["constraints"]],
        prompt_path: str,
        separate_tasks: bool,
        context_tags: typing.List[str] = None,
        only_tags: typing.List[str] = None,
    ):
        super().__init__(steps)
        self._only_tags = only_tags
        self._context_tags = context_tags
        self._prompt_path = prompt_path
        self._prompt = common.load_prompt_from_file(prompt_path)
        self._separate_tasks = separate_tasks
        self._sentence_re = re.compile(
            r"^\s*\*\*\s?sentence\s*(\d+)\s*\*\*\s*$", flags=re.IGNORECASE
        )

    def description(self) -> str:
        return self._prompt

    @property
    def args(self):
        return {
            "prompt_path": self._prompt_path,
            "separate_tasks": self._separate_tasks,
            "context_tags": self._context_tags,
            "only_tags": self._only_tags,
        }

    def _dump_constraints(
        self, constraints: typing.List[data.VanDerAaConstraint]
    ) -> str:
        res = []
        for c in constraints:
            # if c.sentence_id != sentence_id:
            #     continue
            negative = "TRUE" if c.negative else "FALSE"
            tail = ""
            if c.tail is not None:
                tail = c.tail.text
            if self._separate_tasks:
                res.append(f"{negative}\t{c.type}\t{c.head}\t{tail}")
            else:
                res.append(
                    f"{c.sentence_id}\t{negative}\t{c.type}\t{c.head.text}\t{tail}"
                )
        return "\n".join(res)

    @staticmethod
    def _filter_constraints(
        tags_of_interest: typing.List[str],
        constraints: typing.List[data.VanDerAaConstraint],
    ) -> typing.List[data.VanDerAaConstraint]:
        relevant_constraints = []
        for c in constraints:
            if tags_of_interest is None:
                continue
            if c.type.lower() not in tags_of_interest:
                continue
            relevant_constraints.append(c)
        return relevant_constraints

    @staticmethod
    def _dump_actions(document: data.VanDerAaDocument, sentence_id: int) -> str:
        actions: typing.Set[str] = set()
        for c in document.constraints:
            if c.sentence_id != sentence_id:
                continue
            actions.add(c.head.text)
            if c.tail is not None:
                actions.add(c.tail.text)
        return "\n".join(actions)

    def output(self, document: data.VanDerAaDocument) -> str:
        res = []
        relevant_constraints = (
            IterativeVanDerAaRelationListingFormattingStrategy._filter_constraints(
                tags_of_interest=self._only_tags, constraints=document.constraints
            )
        )
        for i, sentence in enumerate(document.sentences):
            if self._separate_tasks:
                res.append(f"Sentence {i}: {sentence}")
        if relevant_constraints is not None and len(relevant_constraints) > 0:
            res.append("Constraints:")
            res.append(self._dump_constraints(relevant_constraints))
        if self._separate_tasks:
            res.append("")
        return "\n".join(res)

    def input(self, document: data.VanDerAaDocument) -> str:
        sentences = "\n".join(
            f"Sentence {i}: {s}" for i, s in enumerate(document.sentences)
        )
        relevant_constraints = (
            IterativeVanDerAaRelationListingFormattingStrategy._filter_constraints(
                tags_of_interest=self._context_tags, constraints=document.constraints
            )
        )
        constraints = [self._dump_constraints(relevant_constraints)]
        # for i in range(len(document.sentences)):
        constraints_as_string = "\n".join(constraints)
        if len(relevant_constraints) > 0:
            constraints_heading = "\nConstraints:\n"
        else:
            constraints_heading = ""
        return f"{sentences}\n{constraints_heading}{constraints_as_string}"

    def parse(self, document: data.VanDerAaDocument, string: str) -> base.ParseResult:
        constraints = []
        current_sentence_id: typing.Optional[int] = None
        for line in string.splitlines(keepends=False):
            if line.strip() == "":
                continue

            match = re.match(self._sentence_re, line)
            if match is not None:
                # new sentence
                current_sentence_id = int(match.group(1))
                continue

            if "\t" not in line:
                # either a header like "Actions:" or "Constraints:",
                # or an action, which we currently do not parse
                continue

            split_line = line.strip().split("\t")
            if self._separate_tasks:
                if len(split_line) == 4:
                    negative, c_type, c_head, c_tail = split_line
                    c_head = data.VanDerAaMention(text=c_head)
                    c_tail = data.VanDerAaMention(text=c_tail)
                elif len(split_line) == 3:
                    negative, c_type, c_head = split_line
                    c_head = data.VanDerAaMention(text=c_head)
                    c_tail = None
                else:
                    print(
                        f'Expected 3 or 4 tab separated values in line "{line}", got {len(split_line)}, skipping line.'
                    )
                    continue
            else:
                if len(split_line) == 5:
                    current_sentence_id, negative, c_type, c_head, c_tail = split_line
                    c_head = data.VanDerAaMention(text=c_head)
                    if c_tail == "":
                        c_tail = None
                    else:
                        c_tail = data.VanDerAaMention(text=c_tail)
                elif len(split_line) == 4:
                    current_sentence_id, negative, c_type, c_head = split_line
                    c_head = data.VanDerAaMention(text=c_head)
                    c_tail = None
                else:
                    print(
                        f'Expected 4 or 5 tab separated values in line "{line}", got {len(split_line)}, skipping line.'
                    )
                    continue
                current_sentence_id = int(current_sentence_id)

            if c_type.strip() == "":
                print(f"Predicted empty type in {line}. Skipping.")
                continue

            if c_tail == "None":
                raise AssertionError()

            constraints.append(
                data.VanDerAaConstraint(
                    sentence_id=current_sentence_id,
                    type=c_type.strip().lower(),
                    head=c_head,
                    tail=c_tail,
                    negative=negative.lower() == "true",
                )
            )
        doc = data.VanDerAaDocument(
            id=document.id,
            name=document.name,
            text=document.text,
            constraints=constraints,
            sentences=document.sentences,
            mentions=document.mentions,
        )
        return base.ParseResult(doc, 0)


class QuishpiMentionListingFormattingStrategy(
    base.BaseFormattingStrategy[data.QuishpiDocument]
):
    def __init__(
        self,
        steps: typing.List[typing.Literal["mentions"]],
        only_tags: typing.Optional[typing.List[str]] = None,
        prompt: str = None,
    ):
        super().__init__(steps)
        if prompt is None:
            prompt = "quishpi/md/long-no-explain.txt"
        self._prompt = prompt
        self._only_tags = only_tags

    def description(self) -> str:
        return common.load_prompt_from_file(self._prompt)

    @property
    def args(self):
        return {"prompt": self._prompt}

    def output(self, document: data.QuishpiDocument) -> str:
        mentions = []
        for m in document.mentions:
            if self._only_tags is not None and m.type.lower() not in self._only_tags:
                continue

            mentions.append(f"{m.type}\t{m.text}")
        if len(mentions) == 0:
            return "No mentions detected."
        return "\n".join(mentions)

    def input(self, document: data.QuishpiDocument) -> str:
        return document.text

    def parse(self, document: data.QuishpiDocument, string: str) -> base.ParseResult:
        mentions: typing.List[data.QuishpiMention] = []

        for line in string.splitlines(keepends=False):
            if "\t" not in line:
                print(f"Skipping non-tab-separated line '{line}'.")
                continue

            split_line = line.split("\t")
            assert 2 <= len(split_line) <= 3, split_line
            if 3 < len(split_line) < 2:
                print(
                    f"Expected two or three tab-separated values, "
                    f"got {len(split_line)} in '{line}' from LLM. Skipping."
                )
                continue

            if len(split_line) == 3:
                mention_type, mention_text, explanation = split_line
                print(f"Explanation for {mention_text} ({mention_type}): {explanation}")
            else:
                mention_type, mention_text = split_line

            mention = data.QuishpiMention(
                type=mention_type.strip().lower(), text=mention_text
            )
            mentions.append(mention)

        doc = data.QuishpiDocument(
            id=document.id, text=document.text, mentions=mentions
        )
        return base.ParseResult(doc, 0)


class IterativeQuishpiMentionListingFormattingStrategy(
    QuishpiMentionListingFormattingStrategy
):
    def __init__(
        self,
        steps: typing.List[typing.Literal["mentions"]],
        tag: str,
        context_tags: typing.List[str],
    ):
        prompt = f"quishpi/md/iterative/{tag.replace(' ', '_')}.txt"
        super().__init__(steps, only_tags=[tag], prompt=prompt)
        self._tag = tag.lower()
        self._context_tags = [t.lower() for t in context_tags]
        self._input_formatter = tags.PetTagFormattingStrategy(
            include_ids=False, only_tags=self._context_tags
        )

    @property
    def args(self):
        return {"tag": self._tag, "context_tags": self._context_tags}

    def description(self) -> str:
        return common.load_prompt_from_file(self._prompt)

    def input(self, document: data.QuishpiDocument) -> str:
        text = document.text
        for i in range(len(document.mentions)):
            mention = document.mentions[len(document.mentions) - i - 1]
            if mention.type.lower() not in self._context_tags:
                continue

            right_index = text.rfind(mention.text)
            while right_index != -1:
                can_replace = True
                # if we are already at the left "edge" of our text, no need to check for an existing tag
                if right_index > 1:
                    # is there an existing tag?
                    can_replace = text[right_index - 2] != ">"
                if not can_replace:
                    right_index = text.rfind(mention.text, 0, right_index)
                    continue

                text = (
                    text[: right_index + len(mention.text)]
                    + f" </{mention.type}>"
                    + text[right_index + len(mention.text) :]
                )
                text = text[:right_index] + f"<{mention.type}> " + text[right_index:]
                break
        return text


class PetRelationListingFormattingStrategy(
    base.BaseFormattingStrategy[data.PetDocument]
):
    def __init__(
        self,
        steps: typing.List[str],
        prompt: str = None,
        only_tags: typing.Optional[typing.List[str]] = None,
        context_tags: typing.Optional[typing.List[str]] = None,
    ):
        super().__init__(steps)
        self._input_formatter = tags.PetTagFormattingStrategy(include_ids=True)
        self._prompt_path = prompt
        self._only_tags = only_tags
        self._context_tags = context_tags
        if self._prompt_path is None:
            self._prompt_path = "pet/re/long.txt"

    def description(self) -> str:
        return common.load_prompt_from_file(self._prompt_path)

    @property
    def args(self):
        return {
            "prompt": self._prompt_path,
            "only_tags": self._only_tags,
            "context_tags": self._context_tags,
        }

    def _format_relations(self, relations: typing.Iterable[data.PetRelation]) -> str:
        res = []
        for r in relations:
            if self._only_tags is not None and r.type.lower() not in self._only_tags:
                continue
            res.append(f"{r.type}\t{r.head_mention_index}\t{r.tail_mention_index}")
        if len(res) == 0:
            return "No relations found."
        return "\n".join(res)

    def output(self, document: data.PetDocument) -> str:
        return self._format_relations(document.relations)

    def input(self, document: data.PetDocument) -> str:
        context_relations = []
        for r in document.relations:
            if self._context_tags is None:
                continue
            if r.type.lower() not in self._context_tags:
                continue
            context_relations.append(r)

        relations = self._format_relations(context_relations)
        text = self._input_formatter.output(document)

        return f"{text}\n\n{relations}"

    def parse(self, document: data.PetDocument, string: str) -> base.ParseResult:
        document = document.copy(clear=["relations"])
        total_errors = 0
        for line in string.splitlines(keepends=False):
            if "\t" not in line:
                print(f"Skipping non-tab-separated line {line}.")
                continue
            split_line = line.split("\t")
            if len(split_line) == 3 or len(split_line) > 4:
                relation_type, head_index, tail_index = split_line
            elif len(split_line) == 4:
                relation_type, head_index, tail_index, explanation = split_line
            else:
                print(
                    f"Expected exactly 3-4 arguments in line {line}, got {len(split_line)}. Skipping."
                )
                total_errors += 1
                continue
            relation_type = relation_type.lower().strip()
            try:
                head_index = int(head_index)
                tail_index = int(tail_index)
            except ValueError:
                total_errors += 1
                continue
            if head_index >= len(document.mentions):
                continue
            if tail_index >= len(document.mentions):
                continue
            document.relations.append(
                data.PetRelation(
                    type=relation_type,
                    head_mention_index=head_index,
                    tail_mention_index=tail_index,
                )
            )
        return base.ParseResult(document, total_errors)


class PetIterativeRelationListingFormattingStrategy(
    PetRelationListingFormattingStrategy
):
    pass


class PetEntityListingFormattingStrategy(base.BaseFormattingStrategy[data.PetDocument]):
    def __init__(self, steps: typing.List[str]):
        super().__init__(steps)
        self._input_formatter = tags.PetTagFormattingStrategy(
            include_ids=True, only_tags=["Activity Data", "Actor"]
        )

    def description(self) -> str:
        return common.load_prompt_from_file("pet/er/long.txt")

    @property
    def args(self):
        return {}

    def output(self, document: data.PetDocument) -> str:
        ret = []
        for e in document.entities:
            ret.append(" ".join([str(i) for i in e.mention_indices]))
        return "\n".join(ret)

    def input(self, document: data.PetDocument) -> str:
        return self._input_formatter.output(document)

    def parse(self, document: data.PetDocument, string: str) -> base.ParseResult:
        document = document.copy(clear=["entities"])
        for line in string.splitlines(keepends=False):
            if " " not in line:
                try:
                    mention_id = int(line)
                    if mention_id >= len(document.mentions):
                        continue
                    mention_ids = []
                except ValueError:
                    print(f"Skipping non space-separated line '{line}'!")
                    continue
            else:
                mention_ids = []
                for i in line.split(" "):
                    try:
                        mention_id = int(i)
                        if mention_id >= len(document.mentions):
                            continue
                        mention_ids.append(mention_id)
                    except ValueError:
                        pass
            mentions = []
            for i in mention_ids:
                if i >= len(document.mentions):
                    continue
                mentions.append(document.mentions[i])
            mention_types = set(m.type for m in mentions)
            if len(mention_types) > 1:
                print(f"Extracted multi-type entity, with mentions {mentions}.")
            document.entities.append(data.PetEntity(mention_indices=tuple(mention_ids)))
        for i, mention in enumerate(document.mentions):
            if any([i in e.mention_indices for e in document.entities]):
                continue
            document.entities.append(data.PetEntity(mention_indices=(i,)))
        return base.ParseResult(document, 0)


class PetMentionListingFormattingStrategy(
    base.BaseFormattingStrategy[data.PetDocument]
):
    def __init__(
        self,
        steps: typing.List[str],
        only_tags: typing.Optional[typing.List[str]] = None,
        generate_descriptions: bool = False,
        prompt: str = None,
    ):
        super().__init__(steps)
        self._generate_descriptions = generate_descriptions
        self._only_tags = only_tags
        if self._only_tags is not None:
            self._only_tags = [t.lower() for t in self._only_tags]
        self._prompt = prompt

    def description(self) -> str:
        if self._prompt is None:
            return common.load_prompt_from_file("pet/md/short_prompt.tx")
        else:
            return common.load_prompt_from_file(self._prompt)

    @property
    def args(self):
        return {
            "only_tags": self._only_tags,
            "generate_descriptions": self._generate_descriptions,
            "prompt": self._prompt,
        }

    def output(self, document: data.PetDocument) -> str:
        formatted_mentions = []
        for i, m in enumerate(document.mentions):
            if self._only_tags is not None and m.type.lower() not in self._only_tags:
                continue

            relation_candidates = [
                r
                for r in document.relations
                if r.head_mention_index == i or r.tail_mention_index == i
            ]
            relevant_relations = [
                r
                for r in relation_candidates
                if r.type.lower() in ["uses", "actor performer", "actor recipient"]
            ]

            description = ""
            if self._generate_descriptions:
                if len(relevant_relations) > 0:
                    relevant_relation = relevant_relations[0]
                    head = document.mentions[relevant_relation.head_mention_index]
                    tail = document.mentions[relevant_relation.tail_mention_index]
                    if relevant_relation.type.lower() == "uses":
                        description = f'"{tail.text(document)}" is an object that is being used in the activity "{head.text(document)}"'
                    if relevant_relation.type.lower() == "actor performer":
                        description = f'"{tail.text(document)}" is an actor that executes the activity "{head.text(document)}"'
                    if relevant_relation.type.lower() == "actor recipient":
                        description = f'"{tail.text(document)}" is an actor that is directly affected by the activity "{head.text(document)}"'
                if description == "":
                    print(
                        f"WARNING: no description generated for '{m.text(document)}'! "
                        f"{len(relation_candidates)} relevant relations "
                        f"(candidates were: {relation_candidates})"
                    )

            first_token = document.tokens[m.token_document_indices[0]]
            formatted_mentions.append(
                f"{m.text(document)}\t{m.type}\t{first_token.sentence_index}\t{description}"
            )
        if len(formatted_mentions) == 0:
            return "No mentions found."
        return "\n".join(formatted_mentions)

    def input(self, document: data.PetDocument) -> str:
        sentences = document.sentences
        text = ""
        for i, sentence in enumerate(sentences):
            text += f"Sentence {i}: "
            text += " ".join(t.text for t in sentence)
            text += "\n"
        return text

    def parse_line(
        self, line: str, document: data.PetDocument
    ) -> typing.Optional[typing.List[data.PetMention]]:
        split_line = line.split("\t")
        split_line = tuple(e for e in split_line if e.strip() != "")

        if len(split_line) < 3 or len(split_line) > 4:
            raise ValueError(
                f"Skipping line {split_line}, as it is not formatted "
                f"properly, expected between 3 and 4 arguments."
            )

        if len(split_line) == 3:
            mention_text, mention_type, sentence_id = split_line
        else:
            mention_text, mention_type, sentence_id, explanation = split_line
            # print(f"Explanation for {mention_text}: {explanation}")

        try:
            sentence_id = int(sentence_id)
        except ValueError:
            raise ValueError(f"Invalid sentence index '{sentence_id}', skipping line.")

        sentence = document.sentences[sentence_id]

        mention_text = mention_text.lower()
        mention_tokens = mention_text.split(" ")

        res = []
        for i, token in enumerate(sentence):
            candidates = sentence[i : i + len(mention_tokens)]
            candidate_text = " ".join(c.text.lower() for c in candidates)

            if candidate_text.lower() != mention_text.lower():
                continue

            res.append(
                data.PetMention(
                    token_document_indices=tuple(
                        c.index_in_document for c in candidates
                    ),
                    type=mention_type.lower().strip(),
                )
            )
        matches_in_sentence = len(res)
        # if matches_in_sentence == 0:
        #     print(f"No match for line with parsed sentence id {sentence_id}: '{line}'")
        # if matches_in_sentence > 1:
        #     print(
        #         f"Multiple matches for line with parsed sentence id {sentence_id}: '{line}'"
        #     )
        return res

    def parse(self, document: data.PetDocument, string: str) -> base.ParseResult:
        parsed_mentions: typing.List[data.PetMention] = []
        num_parse_errors = 0
        for line in string.splitlines(keepends=False):
            line = line.strip()
            if line == "":
                continue
            if "\t" not in line:
                continue

            if re.match("-{3,}", line):
                # print(
                #     "Found divider, will discard all mentions, as they were only candidates"
                # )
                parsed_mentions = []
                continue

            try:
                mentions_from_line = self.parse_line(line, document)
                parsed_mentions.extend(mentions_from_line)
            except Exception:
                num_parse_errors += 1
                # print("Error during parsing of line, skipping line. Error was:")
                # print(traceback.format_exc())

        doc = data.PetDocument(
            id=document.id,
            text=document.text,
            name=document.name,
            category=document.category,
            tokens=[t.copy() for t in document.tokens],
            mentions=parsed_mentions,
            relations=[],
            entities=[],
        )
        return base.ParseResult(doc, num_parse_errors)


class PetActivityListingFormattingStrategy(PetMentionListingFormattingStrategy):
    def __init__(self, steps: typing.List[str]):
        super().__init__(steps, only_tags=["activity"], generate_descriptions=False)

    @property
    def args(self):
        return {}

    def description(self) -> str:
        return common.load_prompt_from_file("pet/md/iterative/activities.txt")


class IterativePetMentionListingFormattingStrategy(PetMentionListingFormattingStrategy):
    def __init__(
        self,
        steps: typing.List[str],
        tag: str,
        context_tags: typing.List[str],
        prompt: str = None,
    ):
        super().__init__(steps, only_tags=[tag], generate_descriptions=False)
        self._tag = tag.lower()
        self._context_tags = [t.lower() for t in context_tags]
        self._input_formatter = tags.PetTagFormattingStrategy(
            include_ids=False, only_tags=self._context_tags
        )
        self._prompt_path = prompt
        if prompt is None:
            self._prompt_path = (
                f"pet/md/iterative/no_explanation/{self._tag.replace(' ', '_')}.txt"
            )

    @property
    def args(self):
        return {"tag": self._tag, "context_tags": self._context_tags}

    def description(self) -> str:
        return common.load_prompt_from_file(self._prompt_path)

    def input(self, document: data.PetDocument) -> str:
        res = []
        # transform to list of sentences with an id in front
        for i, sentence in enumerate(document.sentences):
            sentence_token_indices = {
                token.index_in_document: i for i, token in enumerate(sentence)
            }
            tmp_doc = data.PetDocument(
                id=document.id,
                name=document.name,
                text=document.text,
                category=document.category,
                tokens=sentence,
                mentions=[
                    data.PetMention(
                        type=m.type,
                        token_document_indices=tuple(
                            sentence_token_indices[i] for i in m.token_document_indices
                        ),
                    )
                    for m in document.mentions
                    if document.tokens[m.token_document_indices[0]].sentence_index == i
                ],
                relations=[],
                entities=[],
            )
            res.append(f"Sentence {i}: {self._input_formatter.output(tmp_doc)}")
        return "\n\n".join(res)


class PetActorListingFormattingStrategy(PetMentionListingFormattingStrategy):
    def __init__(self, steps: typing.List[str]):
        super().__init__(steps, only_tags=["actor"], generate_descriptions=False)
        self._input_formatter = tags.PetTagFormattingStrategy(
            include_ids=False, only_tags=["Activity"]
        )

    def description(self) -> str:
        return common.load_prompt_from_file(
            "pet/md/iterative/actors_no_explanation.txt"
        )

    @property
    def args(self):
        return {}

    def input(self, document: data.PetDocument) -> str:
        res = []
        # transform to list of sentences with an id in front
        for i, sentence in enumerate(document.sentences):
            sentence_token_indices = {
                token.index_in_document: i for i, token in enumerate(sentence)
            }
            tmp_doc = data.PetDocument(
                id=document.id,
                name=document.name,
                text=document.text,
                category=document.category,
                tokens=sentence,
                mentions=[
                    data.PetMention(
                        type=m.type,
                        token_document_indices=tuple(
                            sentence_token_indices[i] for i in m.token_document_indices
                        ),
                    )
                    for m in document.mentions
                    if document.tokens[m.token_document_indices[0]].sentence_index == i
                ],
                relations=[],
                entities=[],
            )
            res.append(f"Sentence {i}: {self._input_formatter.output(tmp_doc)}")
        return "\n\n".join(res)


class PetAndListingFormattingStrategy(PetMentionListingFormattingStrategy):
    def __init__(self, steps: typing.List[str]):
        super().__init__(steps, only_tags=["and gateway"], generate_descriptions=False)

    def description(self) -> str:
        return common.load_prompt_from_file("pet/md/iterative/and.txt")

    @property
    def args(self):
        return {}


class PetConditionListingFormattingStrategy(PetMentionListingFormattingStrategy):
    def __init__(self, steps: typing.List[str]):
        super().__init__(
            steps, only_tags=["condition specification"], generate_descriptions=False
        )

    def description(self) -> str:
        return common.load_prompt_from_file("pet/md/iterative/condition.txt")

    @property
    def args(self):
        return {}


class PetDataListingFormattingStrategy(PetMentionListingFormattingStrategy):
    def __init__(self, steps: typing.List[str]):
        super().__init__(
            steps, only_tags=["activity data"], generate_descriptions=False
        )
        self._input_formatter = tags.PetTagFormattingStrategy(
            include_ids=False, only_tags=["Activity", "Actor"]
        )

    def description(self) -> str:
        return common.load_prompt_from_file("pet/md/iterative/data_no_explanation.txt")

    @property
    def args(self):
        return {}

    def input(self, document: data.PetDocument) -> str:
        res = []
        # transform to list of sentences with an id in front
        for i, sentence in enumerate(document.sentences):
            sentence_token_indices = {
                token.index_in_document: i for i, token in enumerate(sentence)
            }
            tmp_doc = data.PetDocument(
                id=document.id,
                name=document.name,
                text=document.text,
                category=document.category,
                tokens=sentence,
                mentions=[
                    data.PetMention(
                        type=m.type,
                        token_document_indices=tuple(
                            sentence_token_indices[i] for i in m.token_document_indices
                        ),
                    )
                    for m in document.mentions
                    if document.tokens[m.token_document_indices[0]].sentence_index == i
                ],
                relations=[],
                entities=[],
            )
            res.append(f"Sentence {i}: {self._input_formatter.output(tmp_doc)}")
        return "\n\n".join(res)


class PetFurtherListingFormattingStrategy(PetMentionListingFormattingStrategy):
    def __init__(self, steps: typing.List[str]):
        super().__init__(
            steps, only_tags=["further specification"], generate_descriptions=False
        )

    def description(self) -> str:
        return common.load_prompt_from_file("pet/md/iterative/further.txt")

    @property
    def args(self):
        return {}


class PetXorListingFormattingStrategy(PetMentionListingFormattingStrategy):
    def __init__(self, steps: typing.List[str]):
        super().__init__(steps, only_tags=["xor gateway"], generate_descriptions=False)

    def description(self) -> str:
        return common.load_prompt_from_file("pet/md/iterative/xor.txt")

    @property
    def args(self):
        return {}


if __name__ == "__main__":

    def main():
        documents = data.PetImporter("../res/data/pet/all.new.jsonl").do_import()
        formatter = PetMentionListingFormattingStrategy(steps=["mentions"])
        for d in documents:
            print("Input:")
            print(formatter.input(d))
            print()
            print("Output:")
            print(formatter.output(d))
            print()
            print("-------")
            print()
            formatter.parse(d, formatter.output(d))

    main()
