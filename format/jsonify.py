import json
import typing

import data
from format import base


class PetJsonifyFormattingStrategy(base.BaseFormattingStrategy[data.PetDocument]):
    def __init__(self, steps: typing.List[str]):
        super().__init__(steps)
        self._dict_exporter = data.PetDictExporter()

    @property
    def args(self):
        return {}

    def description(self) -> str:
        raise NotImplementedError()

    def output(self, document: data.PetDocument) -> str:
        return json.dumps(
            [self._dict_exporter.export_mention(m) for m in document.mentions]
        )

    def input(self, document: data.PetDocument) -> str:
        raise NotImplementedError()

    def parse(self, document: data.PetDocument, string: str) -> data.PetDocument:
        raise NotImplementedError()
