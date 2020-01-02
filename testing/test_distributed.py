from neuraxle.base import Identity
from neuraxle.distributed import ClusteringWrapper, LocalScheduler
from neuraxle.pipeline import Pipeline
from neuraxle.steps.numpy import NumpyConcatenateInnerFeatures


def test_clustering_wrapper():
    p = Pipeline([
        ClusteringWrapper(
            Pipeline([Identity()]),
            scheduler=LocalScheduler(),
            joiner=NumpyConcatenateInnerFeatures(),
            n_jobs=10,
            batch_size=100
        )
    ])
