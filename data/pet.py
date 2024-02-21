import abc
import dataclasses
import json
import typing

from datasets import load_dataset

from data import base


@dataclasses.dataclass
class PetDocument(base.DocumentBase):
    category: str
    name: str
    tokens: typing.List["PetToken"] = dataclasses.field(default_factory=list)
    mentions: typing.List["PetMention"] = dataclasses.field(default_factory=list)
    entities: typing.List["PetEntity"] = dataclasses.field(default_factory=list)
    relations: typing.List["PetRelation"] = dataclasses.field(default_factory=list)

    def copy(
        self,
        clear_mentions: bool = False,
        clear_relations: bool = False,
        clear_entities: bool = False,
    ) -> "PetDocument":
        return PetDocument(
            name=self.name,
            text=self.text,
            id=self.id,
            category=self.category,
            tokens=[t.copy() for t in self.tokens],
            mentions=[] if clear_mentions else [m.copy() for m in self.mentions],
            relations=[] if clear_relations else [r.copy() for r in self.relations],
            entities=[] if clear_entities else [e.copy() for e in self.entities],
        )


@dataclasses.dataclass
class PetMention:
    ner_tag: str
    token_document_indices: typing.List[int] = dataclasses.field(default_factory=list)

    def copy(self) -> "PetMention":
        return PetMention(
            ner_tag=self.ner_tag,
            token_document_indices=[i for i in self.token_document_indices],
        )

    def text(self, document: "PetDocument") -> str:
        return " ".join([document.tokens[i].text for i in self.token_document_indices])

    def to_tuple(self):
        return (self.ner_tag.lower(),) + tuple(sorted(self.token_document_indices))


@dataclasses.dataclass
class PetEntity:
    mention_indices: typing.List[int] = dataclasses.field(default_factory=list)

    def copy(self) -> "PetEntity":
        return PetEntity(mention_indices=[i for i in self.mention_indices])

    def to_tuple(self):
        return tuple(sorted(self.mention_indices))

    def get_tag(self, document: "PetDocument") -> str:
        tags = set(document.mentions[i].ner_tag for i in self.mention_indices)
        if len(tags) > 1:
            print(f"Entity has mentions of mixed ner tags: {tags}")
        return list(tags)[0]


@dataclasses.dataclass
class PetRelation:
    head_mention_index: int
    tail_mention_index: int
    tag: str

    def copy(self) -> "PetRelation":
        return PetRelation(
            head_mention_index=self.head_mention_index,
            tail_mention_index=self.tail_mention_index,
            tag=self.tag,
        )

    def to_tuple(self):
        return self.tag.lower(), self.head_mention_index, self.tail_mention_index


@dataclasses.dataclass
class PetToken:
    text: str
    index_in_document: int
    pos_tag: str
    sentence_index: int

    def copy(self) -> "PetToken":
        return PetToken(
            text=self.text,
            index_in_document=self.index_in_document,
            pos_tag=self.pos_tag,
            sentence_index=self.sentence_index,
        )


class PetJsonExporter:
    def __init__(self, path: str):
        self._dict_exporter = PetDictExporter()
        self._path = path

    def export(self, documents: typing.List[PetDocument]):
        json_lines = []
        for document in documents:
            document_as_json = json.dumps(self._dict_exporter.export_document(document))
            json_lines.append(document_as_json)
        with open(self._path, "w", encoding="utf8") as f:
            f.write("\n".join(json_lines))


class PetDictExporter:
    def export_document(self, document: PetDocument) -> typing.Dict:
        return {
            "text": document.text,
            "name": document.name,
            "id": document.id,
            "category": document.category,
            "tokens": list(map(self.export_token, document.tokens)),
            "mentions": list(map(self.export_mention, document.mentions)),
            "entities": list(map(self.export_entity, document.entities)),
            "relations": list(map(self.export_relation, document.relations)),
        }

    def export_token(self, token: PetToken) -> typing.Dict:
        return {
            "text": token.text,
            "indexInDocument": token.index_in_document,
            "posTag": token.pos_tag,
            "sentenceIndex": token.sentence_index,
        }

    def export_mention(self, mention: PetMention) -> typing.Dict:
        return {
            "nerTag": mention.ner_tag,
            "tokenDocumentIndices": mention.token_document_indices,
        }

    def export_relation(self, relation: PetRelation) -> typing.Dict:
        return {
            "headMentionIndex": relation.head_mention_index,
            "tailMentionIndex": relation.tail_mention_index,
            "tag": relation.tag,
        }

    def export_entity(self, entity: PetEntity) -> typing.Dict:
        return {"mentionIndices": entity.mention_indices}


class OldPetFormatImporter(base.BaseImporter[PetDocument]):
    def __init__(self, file_path: str):
        self._path = file_path

    def do_import(self) -> typing.List[PetDocument]:
        modelhub_dataset = load_dataset(
            "patriziobellan/PET", name="relations-extraction"
        )
        document_names = modelhub_dataset["test"]["document name"]
        original_relations = modelhub_dataset["test"]["relations"]
        return self.read_documents_from_json(
            self._path,
            document_names,
            original_relations,
            modelhub_dataset["test"]["tokens"],
            modelhub_dataset["test"]["sentence-IDs"],
        )

    def read_documents_from_json(
        self,
        file_path: str,
        document_names,
        original_relations,
        original_tokens,
        original_sentences,
    ) -> typing.List[PetDocument]:
        documents = []
        with open(file_path, "r", encoding="utf8") as f:
            for json_line in f:
                json_data = json.loads(json_line)
                documents.append(
                    self._read_document_from_json(
                        json_data, document_names, original_relations
                    )
                )
        return documents

    def _read_document_from_json(
        self, json_data: typing.Dict, document_names, original_relations
    ) -> PetDocument:
        original_index = document_names.index(json_data["id"])
        original_relations = original_relations[original_index]
        tokens = self._read_tokens_from_json(json_data["tokens"])
        mentions = self._read_mentions_from_json(json_data["mentions"], tokens)
        entities = self._read_entities_from_json(json_data["entities"])
        relations = self._read_relations_from_model_hub_data(
            mentions, tokens, json_data["id"], original_relations
        )
        return PetDocument(
            id=json_data["id"],
            name=json_data["id"],
            text=json_data["text"],
            category="",
            tokens=tokens,
            mentions=mentions,
            relations=relations,
            entities=entities,
        )

    def _read_tokens_from_json(
        self, json_tokens: typing.List[typing.Dict]
    ) -> typing.List[PetToken]:
        tokens = []
        for i, json_token in enumerate(json_tokens):
            tokens.append(
                PetToken(
                    text=json_token["text"],
                    pos_tag=json_token["stanza_pos"],
                    index_in_document=i,
                    sentence_index=json_token["sentence_id"],
                )
            )
        return tokens

    def _read_mentions_from_json(
        self, json_mentions: typing.List[typing.Dict], tokens: typing.List[PetToken]
    ) -> typing.List[PetMention]:
        mentions = []
        for json_mention in json_mentions:
            mention = self._read_mention_from_json(json_mention, tokens)
            mentions.append(mention)
        return mentions

    def _read_entities_from_json(
        self, json_entities: typing.List[typing.Dict]
    ) -> typing.List[PetEntity]:
        entities = []
        for json_entity in json_entities:
            entity = self._read_entity_from_json(json_entity)
            entities.append(entity)
        return entities

    def _read_mention_from_json(
        self, json_mention: typing.Dict, tokens: typing.List[PetToken]
    ) -> PetMention:
        sentence_level_token_indices = json_mention["token_indices"]
        sentence_id = json_mention["sentence_id"]
        tokens_by_sentence = []
        last_sentence_id = None
        for token in tokens:
            if last_sentence_id is not token.sentence_index:
                last_sentence_id = token.sentence_index
                tokens_by_sentence.append([])
            tokens_by_sentence[-1].append(token)

        sentence_tokens = [t for t in tokens if t.sentence_index == sentence_id]
        document_level_token_indices = [
            t.index_in_document
            for i, t in enumerate(sentence_tokens)
            if i in sentence_level_token_indices
        ]

        return PetMention(
            ner_tag=json_mention["ner"],
            token_document_indices=document_level_token_indices,
        )

    def _read_entity_from_json(self, json_entity: typing.Dict) -> PetEntity:
        return PetEntity(json_entity["mention_indices"])

    def _read_relations_from_model_hub_data(
        self,
        mentions: typing.List[PetMention],
        tokens: typing.List[PetToken],
        document_id: str,
        original_relations: typing.Dict[str, typing.Any],
    ) -> typing.List[PetRelation]:
        relations = []

        for (
            head_sentence_id,
            head_token_sentence_index,
            tag,
            tail_sentence_id,
            tail_token_sentence_index,
        ) in zip(
            original_relations["source-head-sentence-ID"],
            original_relations["source-head-word-ID"],
            original_relations["relation-type"],
            original_relations["target-head-sentence-ID"],
            original_relations["target-head-word-ID"],
        ):
            mention_start_indices = [m.token_document_indices[0] for m in mentions]

            head_sentence = [t for t in tokens if t.sentence_index == head_sentence_id]
            head_token = head_sentence[head_token_sentence_index]
            head_mention_index = mention_start_indices.index(
                head_token.index_in_document
            )

            # fix known broken data in pet
            if (tail_sentence_id, tail_token_sentence_index) == (
                6,
                16,
            ) and document_id == "doc-2.1":
                tail_sentence_id = 3

            tail_sentence = [t for t in tokens if t.sentence_index == tail_sentence_id]
            tail_token = tail_sentence[tail_token_sentence_index]
            tail_mention_index = mention_start_indices.index(
                tail_token.index_in_document
            )

            relation = PetRelation(
                head_mention_index=head_mention_index,
                tail_mention_index=tail_mention_index,
                tag=tag,
            )
            relations.append(relation)
        return relations


class NewPetFormatImporter(base.BaseImporter[PetDocument]):
    class DictImporter:
        @staticmethod
        def read_tokens_from_dict(
            json_tokens: typing.List[typing.Dict],
        ) -> typing.List[PetToken]:
            tokens = []
            for i, json_token in enumerate(json_tokens):
                tokens.append(
                    PetToken(
                        text=json_token["text"],
                        pos_tag=json_token["posTag"],
                        index_in_document=i,
                        sentence_index=json_token["sentenceIndex"],
                    )
                )
            return tokens

        @staticmethod
        def read_mentions_from_dict(
            json_mentions: typing.List[typing.Dict],
        ) -> typing.List[PetMention]:
            mentions = []
            for json_mention in json_mentions:
                mention = NewPetFormatImporter.DictImporter.read_mention_from_dict(
                    json_mention
                )
                mentions.append(mention)
            return mentions

        @staticmethod
        def read_entities_from_dict(
            json_entities: typing.List[typing.Dict],
        ) -> typing.List[PetEntity]:
            entities = []
            for json_entity in json_entities:
                entity = NewPetFormatImporter.DictImporter.read_entity_from_dict(
                    json_entity
                )
                entities.append(entity)
            return entities

        @staticmethod
        def read_mention_from_dict(json_mention: typing.Dict) -> PetMention:
            return PetMention(
                ner_tag=json_mention["nerTag"],
                token_document_indices=json_mention["tokenDocumentIndices"],
            )

        @staticmethod
        def read_entity_from_dict(json_entity: typing.Dict) -> PetEntity:
            return PetEntity(json_entity["mentionIndices"])

        @staticmethod
        def read_relations_from_dict(
            json_relations: typing.List[typing.Dict],
        ) -> typing.List[PetRelation]:
            relations = []
            for json_relation in json_relations:
                relations.append(
                    NewPetFormatImporter.DictImporter.read_relation_from_dict(
                        json_relation
                    )
                )
            return relations

        @staticmethod
        def read_relation_from_dict(relation_dict: typing.Dict) -> PetRelation:
            head_mention_index = relation_dict["headMentionIndex"]
            tail_mention_index = relation_dict["tailMentionIndex"]
            return PetRelation(
                head_mention_index=head_mention_index,
                tail_mention_index=tail_mention_index,
                tag=relation_dict["tag"],
            )

    def __init__(self, file_path: str):
        self._file_path = file_path

    def do_import(self) -> typing.List[PetDocument]:
        documents: typing.List[PetDocument] = []
        with open(self._file_path, "r", encoding="utf8") as f:
            for json_line in f:
                json_data = json.loads(json_line)
                documents.append(self.read_document_from_json(json_data))
        return documents

    @staticmethod
    def read_document_from_json(json_data: typing.Dict) -> PetDocument:
        mentions = NewPetFormatImporter.DictImporter.read_mentions_from_dict(
            json_data["mentions"]
        )
        entities = NewPetFormatImporter.DictImporter.read_entities_from_dict(
            json_data["entities"]
        )
        relations = NewPetFormatImporter.DictImporter.read_relations_from_dict(
            json_data["relations"]
        )
        tokens = NewPetFormatImporter.DictImporter.read_tokens_from_dict(
            json_data["tokens"]
        )
        document = PetDocument(
            name=json_data["name"],
            text=json_data["text"],
            id=json_data["id"],
            category=json_data["category"],
            tokens=tokens,
            mentions=mentions,
            relations=relations,
            entities=entities,
        )
        return document


if __name__ == "__main__":
    documents = OldPetFormatImporter("res/data/pet/all.jsonl").do_import()
    PetJsonExporter("res/data/pet/all.new.jsonl").export(documents)
    documents = NewPetFormatImporter("res/data/pet/all.new.jsonl").do_import()
