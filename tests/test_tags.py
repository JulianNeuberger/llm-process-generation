import format


def test_ner_to_tag():
    assert format.PetTagFormattingStrategy.ner_to_tag("a", {}) == ("<a>", "</a>")
    assert format.PetTagFormattingStrategy.ner_to_tag("a", {"b": "1"}) == (
        "<a b=1>",
        "</a>",
    )
    assert format.PetTagFormattingStrategy.ner_to_tag("Activity Data", {"id": "3"}) == (
        "<Activity Data id=3>",
        "</Activity Data>",
    )


def test_tag_to_ner():
    assert format.PetTagFormattingStrategy.tag_to_ner("<a>") == "a"
    assert format.PetTagFormattingStrategy.tag_to_ner("<a b=1>") == "a"
    assert (
        format.PetTagFormattingStrategy.tag_to_ner("<Activity Data id=3>")
        == "Activity Data"
    )
