from eval import metrics

import data


def test_constraint_slot_filling_stats():
    document = data.VanDerAaDocument(
        id="id", name="name", text="text", constraints=[], sentences=[]
    )

    true = [
        data.VanDerAaConstraint(
            type="test", negative=False, head="a", tail="b", sentence_id=0
        )
    ]
    pred = [
        data.VanDerAaConstraint(
            type="test", negative=False, head="a", tail="b", sentence_id=0
        )
    ]
    stats = metrics.constraint_slot_filling_stats(
        document, true=true, pred=pred, verbose=False, print_only_tags=[]
    )
    assert stats["test"] == metrics.Stats(num_pred=3, num_gold=3, num_ok=3)

    true = [
        data.VanDerAaConstraint(
            type="test", negative=False, head="a", tail="b", sentence_id=0
        ),
        data.VanDerAaConstraint(
            type="test", negative=False, head="c", tail="d", sentence_id=0
        ),
        data.VanDerAaConstraint(
            type="toast", negative=False, head="c", tail="b", sentence_id=0
        ),
    ]
    pred = [
        data.VanDerAaConstraint(
            type="test", negative=False, head="a", tail="b", sentence_id=0
        )
    ]
    stats = metrics.constraint_slot_filling_stats(
        document, true=true, pred=pred, verbose=False, print_only_tags=[]
    )
    assert stats["test"] == metrics.Stats(num_pred=3, num_gold=6, num_ok=3)
    assert stats["toast"] == metrics.Stats(num_pred=0, num_gold=3, num_ok=0)

    true = [
        data.VanDerAaConstraint(
            type="test", negative=True, head="a", tail="b", sentence_id=0
        )
    ]
    pred = [
        data.VanDerAaConstraint(
            type="test", negative=False, head="a", tail="b", sentence_id=0
        )
    ]
    stats = metrics.constraint_slot_filling_stats(
        document, true=true, pred=pred, verbose=False, print_only_tags=[]
    )
    assert "negation" not in stats
    assert stats["test"] == metrics.Stats(num_pred=3, num_gold=4, num_ok=3)

    true = [
        data.VanDerAaConstraint(
            type="test", negative=True, head="a", tail="b", sentence_id=0
        )
    ]
    pred = [
        data.VanDerAaConstraint(
            type="other", negative=True, head="a", tail="b", sentence_id=0
        )
    ]
    stats = metrics.constraint_slot_filling_stats(
        document, true=true, pred=pred, verbose=False, print_only_tags=[]
    )
    assert "test" in stats
    assert "negation" not in stats
    # make sure the prediction is logged under the best match --> "test"
    assert "other" not in stats
    assert stats["test"] == metrics.Stats(num_pred=4, num_gold=4, num_ok=4)

    true = [
        data.VanDerAaConstraint(
            type="test", negative=False, head="a", tail="b", sentence_id=0
        )
    ]
    pred = [
        data.VanDerAaConstraint(
            type="test", negative=False, head="a", tail="b", sentence_id=1
        )
    ]
    stats = metrics.constraint_slot_filling_stats(
        document, true=true, pred=pred, verbose=False, print_only_tags=[]
    )
    assert stats["test"] == metrics.Stats(num_pred=3, num_gold=3, num_ok=0)
