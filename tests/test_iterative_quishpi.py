import format
import data


def test_input():
    doc = data.QuishpiDocument(
        text="This is a test.",
        id="",
        mentions=[
            data.QuishpiMention(text="is", type="typeA"),
            data.QuishpiMention(text="a test", type="typeB"),
        ],
    )
    formatter = format.IterativeQuishpiMentionListingFormattingStrategy(
        ["mentions"], context_tags=["typeA", "typeB"], tag="action"
    )

    formatted = formatter.input(doc)
    assert formatted == "This <typeA> is </typeA> <typeB> a test </typeB>."

    formatter = format.IterativeQuishpiMentionListingFormattingStrategy(
        ["mentions"], context_tags=["typeA"], tag="action"
    )

    formatted = formatter.input(doc)
    assert formatted == "This <typeA> is </typeA> a test."

    doc = data.QuishpiDocument(
        text="This is a test with repeated test.",
        id="",
        mentions=[
            data.QuishpiMention(text="test", type="typeB"),
            data.QuishpiMention(text="test", type="typeB"),
        ],
    )

    formatter = format.IterativeQuishpiMentionListingFormattingStrategy(
        ["mentions"], context_tags=["typeB"], tag="action"
    )

    formatted = formatter.input(doc)
    assert (
        formatted
        == "This is a <typeB> test </typeB> with repeated <typeB> test </typeB>."
    )
