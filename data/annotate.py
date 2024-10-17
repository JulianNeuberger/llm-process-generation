import collections
import dataclasses
import json
import os
import typing
from format.common import create_PetToken_from_txt

from data.base import TDocument
from data.pet import PetDocument

from datasets import load_dataset

from data import base

CUR_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.normpath(os.path.join(CUR_DIR, "..", "res", "data", "annotate"))

class AnnotateImporter(base.BaseImporter[PetDocument]):
    def __init__(self, file_path: str):
        self._path = os.path.join(DATA_DIR, file_path)
        self.pet_doc = self.do_import()

    def do_import(self) -> typing.List[PetDocument]:
        if self._path[-4:] == ".txt":
            documents = [self.read_documet_from_txt(self._path)]
        else:
            dir_list = os.listdir(self._path)
            txt_list = []
            for dataname in dir_list:
                if dataname[-4:] == "txt":
                    txt_list.append(dataname)
            if not txt_list:
                print("No Txt found in folder")
            else:
                documents = self.read_documents_from_folder(txt_list)
        return documents

    def read_documents_from_folder(self, txt_list: str):
        return NotImplemented

    def read_documet_from_txt(self, file_path: str):
        with open(self._path,"r") as doc:
            text = doc.read().replace("\n"," ")
            name = os.path.splitext(os.path.basename(doc.name))[0]
            print(name)

        return PetDocument(
            id=1,
            name=name,
            text=text,
            category="",
            tokens=create_PetToken_from_txt(self._path),
            mentions="",
            relations="",
            entities="",
        )

    def get_pedDoc(self) -> PetDocument:
        return self.pet_doc




