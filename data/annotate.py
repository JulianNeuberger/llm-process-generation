import collections
import dataclasses
import json
import os
import typing

import data
from format.common import create_PetToken_from_txt

from data.base import TDocument
from data.pet import PetDocument

from datasets import load_dataset

from data import base

CUR_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.normpath(os.path.join(CUR_DIR, "..", "res", "data", "annotate", "pipe", "0_txt"))

class AnnotateImporter(base.BaseImporter[PetDocument]):
    def __init__(self, file_path: str):
        self._path = os.path.join(DATA_DIR, file_path)
        self.pet_doc = self.do_import()

    def do_import(self) -> typing.List[PetDocument]:
        if self._path[-4:] == ".txt":
            documents = [self.read_document_from_txt(self._path,0)]
        else:
            dir_list = os.listdir(self._path)
            txt_list = []
            for dir_name in dir_list:
                if dir_name[-4:] == ".txt":
                    txt_list.append(dir_name)
            if not txt_list:
                print("No Txt found in folder")
            else:
                documents = self.read_documents_from_folder(txt_list)
        return documents

    def read_documents_from_folder(self, txt_list) -> typing.List[PetDocument]:
        pet_list = []
        for i in range(0, len(txt_list)):
            pet_list.append(self.read_document_from_txt(txt_list[i],i))
        return pet_list

    def read_document_from_txt(self, file_path: str, id: int):
        print(file_path)
        path = os.path.join(self._path,file_path)
        print(path)
        with open(path, "r") as doc:
            text = doc.read().replace("\n"," ")
            name = os.path.splitext(os.path.basename(doc.name))[0]
        return PetDocument(
            id=id,
            name=name,
            text=text,
            category="",
            tokens=create_PetToken_from_txt(path),
            mentions="",
            relations="",
            entities="",
        )

    def get_pedDoc(self) -> typing.List[PetDocument]:
        return self.pet_doc

def main():
    #txt = AnnotateImporter("")
    #for x in txt.get_pedDoc():
        #print(x)
    return


