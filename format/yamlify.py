import typing

import yaml

import data
import format


class PetYamlFormattingStrategy(format.BaseFormattingStrategy[data.PetDocument]):
    @staticmethod
    def description() -> str:
        raise NotImplementedError()

    def __init__(
        self, steps: typing.List[typing.Literal["mentions", "entities", "relations"]]
    ):
        super().__init__(steps)
        self._exporter = data.PetDictExporter()

    @staticmethod
    def dump_mention(mention: data.PetMention) -> typing.Dict:
        return {
            "tag": mention.type,
            "indices": " ".join(str(i) for i in mention.token_document_indices),
        }

    @staticmethod
    def load_mention(raw: typing.Dict) -> data.PetMention:
        indices = [int(s) for s in raw["indices"].split(" ")]
        return data.PetMention(ner_tag=raw["tag"], token_document_indices=indices)

    @staticmethod
    def dump_relation(relation: data.PetRelation) -> typing.Dict:
        return {
            "tag": relation.type,
            "headIndex": relation.head_mention_index,
            "tailIndex": relation.tail_mention_index,
        }

    @staticmethod
    def load_relation(raw: typing.Dict) -> data.PetRelation:
        return data.PetRelation(
            tag=raw["tag"],
            head_mention_index=raw["headIndex"],
            tail_mention_index=raw["tailIndex"],
        )

    @staticmethod
    def dump_entity(entity: data.PetEntity) -> typing.Optional[typing.Dict]:
        if len(entity.mention_indices) == 1:
            # save some tokens by only dumping multi-mention entities (others are trivial)
            return None
        return {"indices": " ".join(str(i) for i in entity.mention_indices)}

    @staticmethod
    def load_entity(raw: typing.Dict) -> data.PetEntity:
        raw_indices = raw["indices"]
        if type(raw_indices) is int:
            indices = [raw_indices]
        else:
            indices = [int(s) for s in raw_indices.split(" ")]
        return data.PetEntity(mention_indices=indices)

    def output(self, document: data.PetDocument) -> str:
        content = {}

        if "md" in self._steps:
            content["mentions"] = [self.dump_mention(m) for m in document.mentions]
        if "er" in self._steps:
            entities = [self.dump_entity(e) for e in document.entities]
            entities = [e for e in entities if e is not None]
            content["entities"] = entities
        if "re" in self._steps:
            content["relations"] = [self.dump_relation(r) for r in document.relations]

        return yaml.safe_dump(content)

    def input(self, document: data.PetDocument) -> str:
        content = {"tokens": " ".join([t.text for t in document.tokens])}

        if "md" not in self._steps:
            content["mentions"] = [self.dump_mention(m) for m in document.mentions]
        if "er" not in self._steps and "re" in self._steps:
            entities = [self.dump_entity(e) for e in document.entities]
            entities = [e for e in entities if e is not None]
            content["entities"] = entities

        return yaml.safe_dump(content)

    def parse(self, document: data.PetDocument, string: str) -> data.PetDocument:
        document = document.copy(
            clear_mentions="md" in self._steps,
            clear_entities="er" in self._steps,
            clear_relations="re" in self._steps,
        )

        content = yaml.safe_load(string)

        if "md" in self._steps:
            raw = content["mentions"]
            mentions = [PetYamlFormattingStrategy.load_mention(m) for m in raw]
            document.mentions = mentions

        if "er" in self._steps:
            raw = content["entities"]
            entities = [PetYamlFormattingStrategy.load_entity(e) for e in raw]
            # create single mention entities
            for i, m in enumerate(document.mentions):
                mention_part_of_entity = False
                for e in entities:
                    if i in e.mention_indices:
                        mention_part_of_entity = True
                        break
                if not mention_part_of_entity:
                    entities.append(data.PetEntity(mention_indices=[i]))
            document.entities = entities

        if "re" in self._steps:
            raw = content["relations"]
            relations = [PetYamlFormattingStrategy.load_relation(r) for r in raw]
            document.relations = relations

        return document
