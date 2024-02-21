import dataclasses
import os
import typing

from data import base


@dataclasses.dataclass(frozen=True, eq=True)
class QuishpiMention:
    text: str
    type: str

    def to_tuple(self) -> typing.Tuple:
        return self.type.lower(), self.text.lower()


@dataclasses.dataclass(frozen=True, eq=True)
class QuishpiRelation:
    head: QuishpiMention
    tail: QuishpiMention
    type: str

    def to_tuple(self) -> typing.Tuple:
        return self.type.lower(), self.head.to_tuple(), self.tail.to_tuple()


@dataclasses.dataclass
class QuishpiDocument(base.DocumentBase):
    mentions: typing.List[QuishpiMention]


class QuishpiImporter(base.BaseImporter[QuishpiDocument]):
    def __init__(self, base_dir_path: str, exclude_tags: typing.List[str]):
        self._dir_path = base_dir_path
        self._excluded_tags = [t.lower() for t in exclude_tags]

    def do_import(self) -> typing.List[QuishpiDocument]:
        annotation_path = os.path.join(self._dir_path, "judgeannotations")
        texts_path = os.path.join(self._dir_path, "texts")

        assert os.path.isdir(annotation_path)
        assert os.path.isdir(texts_path)

        annotation_file_names = os.listdir(annotation_path)
        text_file_names = os.listdir(texts_path)

        file_pairs = list(zip(annotation_file_names, text_file_names))

        assert all(
            [os.path.splitext(a)[0] == os.path.splitext(t)[0] for a, t in file_pairs]
        )

        documents: typing.List[QuishpiDocument] = []
        for annotation_file_name, text_file_name in file_pairs:
            annotation_file_path = os.path.join(annotation_path, annotation_file_name)
            with open(annotation_file_path, "r", encoding="utf8") as annotations_file:
                raw_annotations = annotations_file.read()

            text_file_path = os.path.join(texts_path, text_file_name)
            with open(text_file_path, "r", encoding="utf8") as text_file:
                raw_text = text_file.read()

            mentions: typing.Dict[int, QuishpiMention] = {}
            events_to_resolve = []

            for line in raw_annotations.splitlines(keepends=False):
                split_line = line.split("\t")
                annotation_type = split_line[0][0].upper()
                if annotation_type == "T":
                    mention_id, mention = self.mention_from_line(split_line)
                    if mention.type.lower() in self._excluded_tags:
                        continue
                    mentions[mention_id] = mention
                if annotation_type == "A":
                    events_to_resolve.append(split_line)

            for event_line in events_to_resolve:
                event_type, mention_id = event_line[1].split(" ")
                mention_id = mention_id[1:]
                mention_id = int(mention_id)
                old_mention = mentions[mention_id]
                new_mention = QuishpiMention(text=old_mention.text, type=event_type)
                mentions[mention_id] = new_mention

            documents.append(
                QuishpiDocument(
                    text=raw_text,
                    id=os.path.splitext(annotation_file_name)[0],
                    mentions=list(mentions.values()),
                )
            )
        return documents

    @staticmethod
    def relation_from_line(
        split_line: typing.List[str], mentions: typing.Dict[int, QuishpiMention]
    ) -> typing.Tuple[int, QuishpiRelation]:
        relation_id = int(split_line[0][1:])
        relation_type, raw_head, raw_tail = split_line[1].split(" ")
        arg1, head_id = raw_head.split(":")
        arg2, tail_id = raw_tail.split(":")
        assert arg1 == "Arg1"
        assert arg2 == "Arg2"
        head_type = head_id[0]
        tail_type = tail_id[0]
        assert head_type == "T"
        assert tail_type == "T"
        head_id = int(head_id[1:])
        tail_id = int(tail_id[1:])
        head = mentions[head_id]
        tail = mentions[tail_id]
        relation = QuishpiRelation(head=head, tail=tail, type=relation_type)
        return relation_id, relation

    @staticmethod
    def mention_from_line(
        split_line: typing.List[str],
    ) -> typing.Tuple[int, QuishpiMention]:
        mention_id = int(split_line[0][1:])
        mention_type, _, _ = split_line[1].split(" ")
        text = split_line[2]
        return mention_id, QuishpiMention(text=text, type=mention_type)


if __name__ == "__main__":

    def main():
        documents = QuishpiImporter("../res/data/quishpi", exclude_tags=[]).do_import()
        fragment_types = set()
        for d in documents:
            for m in d.mentions:
                fragment_types.add(m.type)
        print("Fragment types: ")
        print("\n".join([t for t in fragment_types]))

    main()
