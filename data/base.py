import abc
import dataclasses
import typing


@dataclasses.dataclass
class DocumentBase:
    id: str
    text: str

    def __add__(self, other):
        raise NotImplementedError()

    def copy(self, clear: typing.List[str]):
        raise NotImplementedError()


TDocument = typing.TypeVar("TDocument", bound=DocumentBase)


class BaseImporter(abc.ABC, typing.Generic[TDocument]):
    def do_import(self) -> typing.List[TDocument]:
        raise NotImplementedError()


@dataclasses.dataclass(frozen=True, eq=True)
class HasType(abc.ABC):
    type: str


class HasCustomMatch:
    def match(self, other: object) -> bool:
        raise NotImplementedError()


TMention = typing.TypeVar("TMention", bound=HasType)


@dataclasses.dataclass
class HasMentions(abc.ABC, typing.Generic[TMention]):
    mentions: typing.List[TMention]


TRelation = typing.TypeVar("TRelation", bound=HasType)


@dataclasses.dataclass
class HasRelations(abc.ABC, typing.Generic[TRelation]):
    relations: typing.List[TRelation]


class SupportsPrettyDump(abc.ABC, typing.Generic[TDocument]):
    def pretty_dump(self, document: TDocument, human_readable: bool = False) -> str:
        raise NotImplementedError()
