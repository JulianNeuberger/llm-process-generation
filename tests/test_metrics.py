from eval import metrics

import data


def test_constraint_slot_filling_stats():
    document = data.VanDerAaDocument(id="id", name="name", text="text", constraints=[])

    true = [data.VanDerAaConstraint(type="test", negative=False, head="a", tail="b")]
    pred = [data.VanDerAaConstraint(type="test", negative=False, head="a", tail="b")]
    stats = metrics.constraint_slot_filling_stats(
        document, true=true, pred=pred, verbose=False
    )
    assert stats["test"] == metrics.Stats(num_pred=3, num_gold=3, num_ok=3)

    true = [
        data.VanDerAaConstraint(type="test", negative=False, head="a", tail="b"),
        data.VanDerAaConstraint(type="test", negative=False, head="c", tail="d"),
        data.VanDerAaConstraint(type="toast", negative=False, head="c", tail="b"),
    ]
    pred = [data.VanDerAaConstraint(type="test", negative=False, head="a", tail="b")]
    stats = metrics.constraint_slot_filling_stats(
        document, true=true, pred=pred, verbose=False
    )
    assert stats["test"] == metrics.Stats(num_pred=3, num_gold=6, num_ok=3)
    assert stats["toast"] == metrics.Stats(num_pred=0, num_gold=3, num_ok=0)

    true = [data.VanDerAaConstraint(type="test", negative=True, head="a", tail="b")]
    pred = [data.VanDerAaConstraint(type="test", negative=False, head="a", tail="b")]
    stats = metrics.constraint_slot_filling_stats(
        document, true=true, pred=pred, verbose=False
    )
    assert "negation" in stats
    assert stats["negation"] == metrics.Stats(num_pred=0, num_gold=4, num_ok=0)
    assert stats["test"] == metrics.Stats(num_pred=3, num_gold=0, num_ok=3)

    true = [data.VanDerAaConstraint(type="test", negative=True, head="a", tail="b")]
    pred = [data.VanDerAaConstraint(type="other", negative=True, head="a", tail="b")]
    stats = metrics.constraint_slot_filling_stats(
        document, true=true, pred=pred, verbose=False
    )
    assert "test" not in stats
    assert "other" not in stats
    assert stats["negation"] == metrics.Stats(num_pred=4, num_gold=4, num_ok=4)
