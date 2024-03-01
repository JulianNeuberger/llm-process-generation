import abc
import typing

TDocument = typing.TypeVar("TDocument")


class BaseFormattingStrategy(abc.ABC, typing.Generic[TDocument]):
    def __init__(self, steps: typing.List[str]):
        assert len(steps) > 0
        self._steps = steps

    @property
    def steps(self):
        return self._steps

    def description(self) -> str:
        raise NotImplementedError()

    def output(self, document: TDocument) -> str:
        """
        Formats document in expected output format, e.g. useful for
        few shot settings, to provide correct examples.

        :param document: document to format
        :return: formatted document as string
        """
        raise NotImplementedError()

    def input(self, document: TDocument) -> str:
        """
        Formats document so it can be used as input for prompts.

        :param document: document to format
        :return: formatted document as string
        """
        raise NotImplementedError()

    def parse(self, document: TDocument, string: str) -> TDocument:
        """
        Parses output of LLM, inverse of method output

        :param document: the original document, used if LLM was given
        information about e.g. mentions
        :param string: LLM output
        :return: copy of given document, with added information,
        i.e. mentions, entities, and/or relations
        """
        raise NotImplementedError()
