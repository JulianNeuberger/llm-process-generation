import data


def test_import():
    importer = data.VanDerAaImporter("../res/data/van-der-aa/datacollection.csv")
    docs = importer.do_import()

    assert len(docs) == 17
    assert len(docs[0].sentences) == 30
    assert len(docs[0].constraints) == 33
