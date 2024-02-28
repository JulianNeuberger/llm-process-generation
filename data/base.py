import abc
import dataclasses
import typing


@dataclasses.dataclass
class DocumentBase:
    id: str
    text: str


TDocument = typing.TypeVar("TDocument", bound=DocumentBase)


class BaseImporter(abc.ABC, typing.Generic[TDocument]):
    def do_import(self) -> typing.List[TDocument]:
        raise NotImplementedError()


@dataclasses.dataclass(frozen=True, eq=True)
class HasType(abc.ABC):
    type: str


TMention = typing.TypeVar("TMention", bound=HasType)


@dataclasses.dataclass
class HasMentions(abc.ABC, typing.Generic[TMention]):
    mentions: typing.List[TMention]


TRelation = typing.TypeVar("TRelation", bound=HasType)


@dataclasses.dataclass
class HasRelations(abc.ABC, typing.Generic[TRelation]):
    relations: typing.List[TRelation]


class SupportsPrettyDump(abc.ABC, typing.Generic[TDocument]):
    def pretty_dump(self, document: TDocument) -> str:
        raise NotImplementedError()
