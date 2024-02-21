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
