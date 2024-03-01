import data


def test_pet():
    doc1 = data.PetDocument(
        id="1",
        name="1",
        text="",
        category="",
        tokens=[
            data.PetToken("This", pos_tag="", sentence_index=0, index_in_document=0),
            data.PetToken("is", pos_tag="", sentence_index=0, index_in_document=1),
            data.PetToken("a", pos_tag="", sentence_index=0, index_in_document=2),
            data.PetToken("test", pos_tag="", sentence_index=0, index_in_document=3),
            data.PetToken(".", pos_tag="", sentence_index=0, index_in_document=4),
            data.PetToken("And", pos_tag="", sentence_index=1, index_in_document=5),
            data.PetToken("a", pos_tag="", sentence_index=1, index_in_document=6),
            data.PetToken("second", pos_tag="", sentence_index=1, index_in_document=7),
            data.PetToken("one", pos_tag="", sentence_index=1, index_in_document=8),
            data.PetToken(".", pos_tag="", sentence_index=1, index_in_document=9),
        ],
        mentions=[data.PetMention("A", (2, 3)), data.PetMention("B", (4,))],
        entities=[data.PetEntity((0,))],
        relations=[data.PetRelation("R", 1, 0)],
    )

    doc2 = data.PetDocument(
        id="1",
        name="1",
        text="",
        category="",
        tokens=[
            data.PetToken("This", pos_tag="", sentence_index=0, index_in_document=0),
            data.PetToken("is", pos_tag="", sentence_index=0, index_in_document=1),
            data.PetToken("a", pos_tag="", sentence_index=0, index_in_document=2),
            data.PetToken("test", pos_tag="", sentence_index=0, index_in_document=3),
            data.PetToken(".", pos_tag="", sentence_index=0, index_in_document=4),
            data.PetToken("And", pos_tag="", sentence_index=1, index_in_document=5),
            data.PetToken("a", pos_tag="", sentence_index=1, index_in_document=6),
            data.PetToken("second", pos_tag="", sentence_index=1, index_in_document=7),
            data.PetToken("one", pos_tag="", sentence_index=1, index_in_document=8),
            data.PetToken(".", pos_tag="", sentence_index=1, index_in_document=9),
        ],
        mentions=[
            data.PetMention("B", (7, 8)),
            data.PetMention("B", (5,)),
            data.PetMention("A", (2, 3)),
            data.PetMention("B", (4,)),
        ],
        entities=[
            data.PetEntity((2,)),
            data.PetEntity(
                (
                    0,
                    1,
                )
            ),
        ],
        relations=[
            data.PetRelation("R", 0, 1),
            data.PetRelation("R", 3, 2),
            data.PetRelation("L", 3, 2),
        ],
    )

    added = doc1 + doc2
    assert len(added.mentions) == 4
    assert len(added.entities) == 2, added.entities
    assert added.entities[1].mention_indices == (2, 3)
    assert len(added.relations) == 3
    assert added.relations[2].head_mention_index == 1
    assert added.relations[2].tail_mention_index == 0


def test_quishpi():
    doc1 = data.QuishpiDocument(
        id="",
        text="",
        mentions=[
            data.QuishpiMention("A", "first mention"),
            data.QuishpiMention("A", "second mention"),
            data.QuishpiMention("A", "third"),
        ],
    )

    doc2 = data.QuishpiDocument(
        id="",
        text="",
        mentions=[
            data.QuishpiMention("A", "first mention"),
            data.QuishpiMention("A", "fourth"),
        ],
    )

    added = doc1 + doc2

    assert len(added.mentions) == 4


def test_van_der_aa():
    doc1 = data.VanDerAaDocument(
        id="",
        text="",
        name="",
        constraints=[
            data.VanDerAaConstraint("A", "arg1", "arg2", negative=True, sentence_id=0),
            data.VanDerAaConstraint("C", "arg2", "arg3", negative=False, sentence_id=1),
            data.VanDerAaConstraint("A", "arg3", "arg2", negative=False, sentence_id=1),
        ],
        sentences=["A", "B"],
    )

    doc2 = data.VanDerAaDocument(
        id="",
        text="",
        name="",
        constraints=[
            # same
            data.VanDerAaConstraint("A", "arg1", "arg2", negative=True, sentence_id=0),
            # different
            data.VanDerAaConstraint("C", "arg3", "arg2", negative=False, sentence_id=0),
            data.VanDerAaConstraint("Z", "arg3", "arg2", negative=False, sentence_id=1),
            data.VanDerAaConstraint(
                "X", "arg99", "arg89", negative=False, sentence_id=1
            ),
        ],
        sentences=["A", "B"],
    )

    added = doc1 + doc2

    assert len(added.constraints) == 6
    assert len(added.sentences) == 2
