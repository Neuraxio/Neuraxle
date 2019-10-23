import numpy as np
from sklearn.decomposition import PCA

from neuraxle.base import Identity
from neuraxle.hyperparams.distributions import RandInt
from neuraxle.hyperparams.space import HyperparameterSpace
from neuraxle.pipeline import Pipeline
from neuraxle.steps.numpy import MultiplyByN
from neuraxle.steps.sklearn import SKLearnWrapper


def main():
    p = Pipeline([
        ('step1', MultiplyByN()),
        ('step2', MultiplyByN()),
        Pipeline([
            Identity(),
            Identity(),
            SKLearnWrapper(PCA(n_components=4))
        ])
    ])

    p.set_hyperparams_space(HyperparameterSpace({
        'step1__multiply_by': RandInt(42, 50),
        'step2__multiply_by': RandInt(-10, 0),
        'Pipeline__SKLearnWrapper_PCA__n_components': RandInt(2, 3)
    }))

    samples = p.get_hyperparams_space().rvs()
    p.set_hyperparams(samples)

    samples = p.get_hyperparams()
    assert 42 <= samples['step1__multiply_by'] <= 50
    assert -10 <= samples['step2__multiply_by'] <= 0
    assert samples['Pipeline__SKLearnWrapper_PCA__n_components'] in [2, 3]
    assert p['Pipeline']['SKLearnWrapper_PCA'].get_wrapped_sklearn_predictor().n_components in [2, 3]

    # JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ

if __name__ == "__main__":
    main()
