from neuraxle.base import Identity
from neuraxle.pipeline import Pipeline


def test_apply_to_children():
    p = Pipeline([
        Identity(),
        Identity()
    ])

    p.apply()
