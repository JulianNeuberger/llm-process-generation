import json
import typing

import data
from format import base, tags


class PetJsonifyFormattingStrategy(base.BaseFormattingStrategy[data.PetDocument]):
    def __init__(self, steps: typing.List[str]):
        super().__init__(steps)
        self._dict_exporter = data.PetDictExporter()
        self._dict_importer = data.PetImporter.DictImporter()
        self._tag_formatter = tags.PetTagFormattingStrategy(include_ids=True)

    @property
    def args(self):
        return {}

    def description(self) -> str:
        raise NotImplementedError()

    def output(self, document: data.PetDocument) -> str:
        out = {}
        if "mentions" in self.steps:
            out["mentions"] = [
                self._dict_exporter.export_mention(m) for m in document.mentions
            ]
        if "entities" in self.steps:
            out["entities"] = [
                self._dict_exporter.export_entity(e) for e in document.entities
            ]
        if "relations" in self.steps:
            out["relations"] = [
                self._dict_exporter.export_relation(r) for r in document.relations
            ]
        return json.dumps(out)

    def input(self, document: data.PetDocument) -> str:
        if "mentions" in self.steps:
            return document.text
        return self._tag_formatter.output(document)

    def parse(self, document: data.PetDocument, string: str) -> data.PetDocument:
        document = document.copy(clear=self.steps)
        parsed = json.loads(string)
        if "mentions" in self.steps:
            document.mentions = self._dict_importer.read_mentions_from_dict(
                parsed["mentions"]
            )
        if "entities" in self.steps:
            document.entities = self._dict_importer.read_entities_from_dict(
                parsed["entities"]
            )
        if "relations" in self.steps:
            document.relations = self._dict_importer.read_relations_from_dict(
                parsed["relations"]
            )
        return document
