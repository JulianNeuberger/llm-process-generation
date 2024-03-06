import data
import format
from format import common


def test_combined_output():
    doc = data.VanDerAaDocument(
        id="1",
        text="This is a first sentence.\nAnd a second one.",
        name="1",
        sentences=["This is a first sentence", "And a second one."],
        constraints=[
            data.VanDerAaConstraint(
                type="A", head="this", tail="first", negative=False, sentence_id=0
            ),
            data.VanDerAaConstraint(
                type="B", head="this", tail="sentence", negative=True, sentence_id=0
            ),
            data.VanDerAaConstraint(
                type="C", head="a", tail="one", negative=False, sentence_id=1
            ),
            data.VanDerAaConstraint(
                type="D", head="second", tail=None, negative=False, sentence_id=1
            ),
        ],
    )

    formatter = format.VanDerAaRelationListingFormattingStrategy(
        steps=["constraints"],
        prompt_path="van-der-aa/re/step-wise.txt",
        separate_tasks=True,
    )

    output = formatter.output(doc)
    assert "Actions:" in output
    assert "Constraints:" in output
    assert "** Sentence 0 **" in output
    assert "** Sentence 1 **" in output
    assert "FALSE\tA\tthis\tfirst" in output
    assert "TRUE\tB\tthis\tsentence" in output
    assert "FALSE\tC\ta\tone" in output
    assert "FALSE\tD\tsecond\t" in output
    assert "FALSE\tD\tsecond\tNone" not in output

    formatter = format.VanDerAaRelationListingFormattingStrategy(
        steps=["constraints"],
        prompt_path="van-der-aa/re/step-wise.txt",
        separate_tasks=False,
    )

    output = formatter.output(doc)
    assert "Actions:" not in output
    assert "Constraints:" not in output
    assert "** Sentence 0 **" not in output
    assert "** Sentence 1 **" not in output
    assert "0\tFALSE\tA\tthis\tfirst" in output
    assert "0\tTRUE\tB\tthis\tsentence" in output
    assert "1\tFALSE\tC\ta\tone" in output
    assert "1\tFALSE\tD\tsecond\t" in output
    assert "FALSE\tD\tsecond\tNone" not in output


def test_parse_output_van_der_aa():
    input_doc = data.VanDerAaImporter(
        "../res/data/van-der-aa/datacollection.csv"
    ).do_import()[0]

    formatter = format.VanDerAaRelationListingFormattingStrategy(
        steps=["constraints"],
        prompt_path="van-der-aa/re/step-wise.txt",
        separate_tasks=True,
    )

    output = formatter.output(input_doc)
    parsed_doc = formatter.parse(input_doc, output)

    for c in input_doc.constraints:
        assert c in parsed_doc.constraints, f"{c} missing in {parsed_doc.constraints}"


def test_parse_output_quishpi():
    input_doc = data.VanDerAaImporter("../res/data/quishpi/csv").do_import()[0]

    formatter = format.VanDerAaRelationListingFormattingStrategy(
        steps=["constraints"],
        prompt_path="quishpi/re/hand-crafted-task-separation-examples.txt",
        separate_tasks=True,
    )

    output = formatter.output(input_doc)
    parsed_doc = formatter.parse(input_doc, output)

    for c in input_doc.constraints:
        assert c in parsed_doc.constraints, f"{c} missing in {parsed_doc.constraints}"


def test_parse_handwritten_quishpi():
    expected_doc = data.VanDerAaDocument(
        id="test",
        text="",
        name="test",
        sentences=[],
        constraints=[
            data.VanDerAaConstraint(
                type="init",
                head="submit paper",
                tail=None,
                negative=False,
                sentence_id=0,
            ),
            data.VanDerAaConstraint(
                type="precedence",
                head="sign contract",
                tail="advertise product",
                negative=False,
                sentence_id=1,
            ),
            data.VanDerAaConstraint(
                type="response",
                head="notify manager",
                tail="reject request",
                negative=True,
                sentence_id=5,
            ),
        ],
    )

    quishpi = common.load_prompt_from_file(
        "quishpi/re/hand-crafted-task-separation-examples.txt"
    )
    quishpi = quishpi.split("## Output")[1]
    quishpi = quishpi.split("# Notes")[0]
    print(quishpi)
    formatter = format.VanDerAaRelationListingFormattingStrategy(
        steps=["constraints"],
        prompt_path="quishpi/re/hand-crafted-task-separation-examples.txt",
        separate_tasks=True,
    )
    parsed = formatter.parse(expected_doc, quishpi)
    for c in expected_doc.constraints:
        assert c in parsed.constraints


def test_parse_handwritten_van_der_aa():
    expected_doc = data.VanDerAaDocument(
        id="test",
        text="",
        name="test",
        sentences=[],
        constraints=[
            data.VanDerAaConstraint(
                type="init",
                head="submit paper",
                tail=None,
                negative=False,
                sentence_id=0,
            ),
            data.VanDerAaConstraint(
                type="precedence",
                head="sign contract",
                tail="advertise product",
                negative=False,
                sentence_id=1,
            ),
            data.VanDerAaConstraint(
                type="response",
                head="notify manager",
                tail="reject request",
                negative=True,
                sentence_id=5,
            ),
        ],
    )

    quishpi = common.load_prompt_from_file(
        "quishpi/re/hand-crafted-task-separation-examples.txt"
    )
    quishpi = quishpi.split("## Output")[1]
    quishpi = quishpi.split("# Notes")[0]
    print(quishpi)
    formatter = format.VanDerAaRelationListingFormattingStrategy(
        steps=["constraints"],
        prompt_path="quishpi/re/hand-crafted-task-separation-examples.txt",
        separate_tasks=True,
    )
    parsed = formatter.parse(expected_doc, quishpi)
    for c in expected_doc.constraints:
        assert c in parsed.constraints
