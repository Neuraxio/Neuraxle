import os
import random
from typing import List

import numpy as np
import pytest
from neuraxle.base import ExecutionContext as CX
from neuraxle.hyperparams.distributions import (Choice, DistributionMixture,
                                                HPSampledValue,
                                                HyperparameterDistribution,
                                                LogNormal, LogUniform, Normal,
                                                Quantized, RandInt, Uniform)
from neuraxle.hyperparams.space import (HyperparameterSamples,
                                        HyperparameterSpace)
from neuraxle.metaopt.auto_ml import (AutoML, BaseHyperparameterOptimizer,
                                      ControlledAutoML)
from neuraxle.metaopt.callbacks import ScoringCallback
from neuraxle.metaopt.data.aggregates import (MetricResults, Round, Trial,
                                              TrialSplit)
from neuraxle.metaopt.data.reporting import RoundReport
from neuraxle.metaopt.data.vanilla import (AutoMLContext,
                                           HyperparameterSamplerStub,
                                           MetricResultsDataclass,
                                           RoundDataclass, ScopedLocation,
                                           TrialDataclass, TrialSplitDataclass,
                                           VanillaHyperparamsRepository)
from neuraxle.metaopt.hyperopt.tpe import (TreeParzenEstimator,
                                           _DividedMixturesFactory,
                                           _DividedTPEPosteriors)
from neuraxle.metaopt.validation import (GridExplorationSampler,
                                         ValidationSplitter)
from neuraxle.pipeline import Pipeline
from neuraxle.steps.numpy import AddN
from sklearn.metrics import mean_squared_error

N_TRIALS = 40


def _avg(seq: List):
    if len(seq) == 0:
        return None
    return sum(seq) / len(seq)


distributions: List[HyperparameterDistribution] = [
    LogNormal(log2_space_mean=1.0, log2_space_std=0.5, hard_clip_min=0, hard_clip_max=6),
    Choice(choice_list=[0, 1.5, 2, 3.5, 4, 5, 6]),
    LogUniform(min_included=1.0, max_included=5.0),
    Normal(mean=3.0, std=2.0, hard_clip_min=0, hard_clip_max=6),
    Quantized(Uniform(0, 10)),
    Uniform(0, 6),
]

# @pytest.mark.parametrize("reduce_func", [min, _avg])


@pytest.mark.parametrize("add_range", distributions)
def test_tpe(tmpdir, add_range: HyperparameterDistribution, reduce_func=_avg):
    np.random.seed(42)
    random.seed(42)

    # Given
    pipeline = Pipeline([
        AddN(0.).set_hyperparams_space(HyperparameterSpace({'add': add_range})),
        AddN(0.).set_hyperparams_space(HyperparameterSpace({'add': RandInt(1, 2)})),
    ])
    half_trials = int(N_TRIALS / 2)
    tpe: BaseHyperparameterOptimizer = TreeParzenEstimator(
        number_of_initial_random_step=half_trials,
        quantile_threshold=0.3,
        number_good_trials_max_cap=25,
        number_possible_hyperparams_candidates=100,
        use_linear_forgetting_weights=False,
        number_recent_trials_at_full_weights=25
    )
    grid: BaseHyperparameterOptimizer = GridExplorationSampler(N_TRIALS)

    grid_scores = _score_meta_optimizer(grid, pipeline, tmpdir)
    tpe_scores = _score_meta_optimizer(tpe, pipeline, tmpdir)

    mean_compared_grid_score = reduce_func(grid_scores[half_trials:])
    mean_compared_tpe_score = reduce_func(tpe_scores[half_trials:])

    assert mean_compared_tpe_score <= mean_compared_grid_score


def _score_meta_optimizer(meta_optimizer: BaseHyperparameterOptimizer, pipeline, tmpdir):
    expected_output_mult = 3.5
    optim_scores = _score_trials(
        expected_output_mult,
        pipeline,
        meta_optimizer,
        os.path.join(tmpdir, meta_optimizer.__class__.__name__)
    )
    return optim_scores


def _score_trials(
    expected_output_mult,
    pipeline,
    hyperparams_optimizer: BaseHyperparameterOptimizer,
    tmpdir: str
):
    hp_repository: VanillaHyperparamsRepository = VanillaHyperparamsRepository(str(tmpdir))
    n_epochs = 1
    auto_ml: ControlledAutoML = AutoML(
        pipeline=pipeline,
        hyperparams_optimizer=hyperparams_optimizer,
        validation_splitter=ValidationSplitter(0.5),
        scoring_callback=ScoringCallback(mean_squared_error, higher_score_is_better=False, name='mse'),
        n_trials=N_TRIALS,
        refit_best_trial=True,
        epochs=n_epochs,
        hyperparams_repository=hp_repository,
        continue_loop_on_error=False
    )

    # When
    data_inputs = np.array([0, 0])
    # Fitting and optimizing a multiplier problem with a model make to add...
    expected_outputs = expected_output_mult * np.ones_like(data_inputs)
    auto_ml.fit(data_inputs=data_inputs, expected_outputs=expected_outputs)

    # Then
    trials_validation_scores: List[float] = auto_ml.report.list_avg_validation_scores()
    return trials_validation_scores


def test_divided_posteriors_rvs_good_is_in_range():
    good_trials = RandInt(1, 2)
    bad_trials = RandInt(101, 102)
    divider: _DividedTPEPosteriors = _DividedTPEPosteriors(good_trials, bad_trials)

    goods: List[int] = []
    for i in range(100):
        goods.append(divider.rvs_good())

    assert good_trials.min() <= min(goods)
    assert max(goods) <= good_trials.max()


def test_divided_posteriors_ratio_is_ok():
    good_trials = Normal(mean=1, std=0.5, hard_clip_min=0, hard_clip_max=3)
    bad_trials = Normal(mean=2, std=0.5, hard_clip_min=0, hard_clip_max=3)
    divider: _DividedTPEPosteriors = _DividedTPEPosteriors(good_trials, bad_trials)
    gmm: DistributionMixture = DistributionMixture.build_gaussian_mixture(
        distribution_amplitudes=[0.5, 0.5],
        means=[1, 2],
        stds=[0.5, 0.5],
        distributions_mins=[0, 0],
        distributions_max=[3, 3],
    )

    for i in range(100):
        good_hyperparam, proba_ratio = divider.rvs_good_with_pdf_division_proba()

        assert proba_ratio > 0.0
        if good_hyperparam > gmm.mean():
            assert proba_ratio < 1.0
        else:
            assert proba_ratio >= 1.0
    assert gmm.mean() == 1.5


class TrialBuilder:

    def __init__(self, round_scope: Round) -> None:
        self.round_scope: Round = round_scope

    def with_optimizer(
        self,
        hp_optimizer: BaseHyperparameterOptimizer = None,
        hp_space: HyperparameterSpace = None,
    ) -> 'TrialBuilder':
        self.round_scope.with_optimizer(hp_optimizer, hp_space)
        self.round_scope.save()
        return self

    def add_trial_from_space_rvs(
        self,
        score: float,
        metric_name: str,
        is_higher_score_better: bool,
        hyperparams_samples: HyperparameterSamples = None,
    ) -> 'TrialBuilder':

        if hyperparams_samples is not None:
            self.round_scope.hp_optimizer = HyperparameterSamplerStub(hyperparams_samples)

        with self.round_scope.new_rvs_trial() as trial:
            trial: Trial = trial

            with trial.new_validation_split() as trial_split:
                trial_split: TrialSplit = trial_split

                with trial_split.managed_metric(metric_name, is_higher_score_better) as metric:
                    metric: MetricResults = metric

                    metric.add_train_result(score)
                    metric.add_valid_result(score)

        self.round_scope.save()
        return self

    def add_many_trials_from_space_rvs(
        self,
        scores: List[float],
        metric_name: str,
        is_higher_score_better: bool,
        hp_samples: List[HyperparameterSamples] = None,
    ) -> 'TrialBuilder':
        if hp_samples is None:
            hp_samples = [None] * len(scores)
        for score, hp in zip(scores, hp_samples):
            self.add_trial_from_space_rvs(score, metric_name, is_higher_score_better, hp)
        return self

    def build(self) -> Round:
        return self.round_scope.save()


def test_divided_mixtures_factory():
    hp_space = HyperparameterSpace([('a', Uniform(0, 1)), ('b', LogUniform(1, 2))])
    round_scope: Round = Round.dummy()
    round_scope.with_optimizer(None, hp_space).with_metric('mse')
    round_scope = TrialBuilder(round_scope).add_many_trials_from_space_rvs(
        scores=[0.0, 0.1, 0.9, 1.0],
        metric_name='mse',
        is_higher_score_better=False,
        hp_samples=[
            HyperparameterSamples([('a', 0.1), ('b', 1.1)]),
            HyperparameterSamples([('a', 0.9), ('b', 1.1)]),
            HyperparameterSamples([('a', 0.1), ('b', 1.9)]),
            HyperparameterSamples([('a', 0.9), ('b', 1.9)]),
        ]
    ).build()

    dmf: _DividedMixturesFactory = _DividedMixturesFactory(
        quantile_threshold=0.5,
        number_good_trials_max_cap=10,
        use_linear_forgetting_weights=False,
        number_recent_trials_at_full_weights=10
    )
    hps_names = List[str]
    divided_distributions = List['_DividedTPEPosteriors']

    hps_names, divided_distributions = dmf.create_from(round_scope)
    a_good_trials = divided_distributions[0].good_trials
    a_bad_trials = divided_distributions[0].bad_trials
    b_good_trials = divided_distributions[1].good_trials
    b_bad_trials = divided_distributions[1].bad_trials

    assert a_good_trials.mean() == 0.5
    assert a_bad_trials.mean() == 0.5
    assert b_good_trials.mean() < 1.5
    assert b_bad_trials.mean() > 1.5

    best_ratios = []
    best_samples = []
    for i, dist in enumerate(divided_distributions):
        best_ratio = 0
        best_hp_sample = None
        for _ in range(100):

            hp_sample, ratio = dist.rvs_good_with_pdf_division_proba()
            if ratio > best_ratio or best_hp_sample is None:
                best_ratio = ratio
                best_hp_sample = hp_sample

        best_ratios.append(best_ratio)
        best_samples.append(best_hp_sample)

    assert best_ratios[0] == 1.0
    assert best_ratios[1] > 1.0
    assert 1.0 <= best_samples[1] < 1.2


@pytest.mark.parametrize("dist", [
    LogNormal(log2_space_mean=1.0, log2_space_std=0.5, hard_clip_min=0, hard_clip_max=6),
    LogNormal(log2_space_mean=1.0, log2_space_std=0.5, hard_clip_min=0),
    Choice(choice_list=[0, 1, 2, 3, 4, 5, 6]),
    LogUniform(min_included=1.0, max_included=5.0),
    Normal(mean=3.0, std=2.0, hard_clip_min=0, hard_clip_max=6),
    Normal(mean=3.0, std=2.0),
    Quantized(Uniform(0, 10)),
    Uniform(0, 6),
])
def test_icdf_is_within_tolerance_value(dist: HyperparameterDistribution):
    tolerance = 2e-5

    # if isinstance(dist, ContinuousHyperparameterDistribution):
    #     dist.min = dist._pseudo_min
    #     dist.max = dist._pseudo_max
    _min = dist.min() if dist.is_discrete() else dist._pseudo_min()
    _max = dist.max() if dist.is_discrete() else dist._pseudo_max()

    # create x range
    if dist.is_discrete():
        x_range = dist.values()
    else:
        x_range = np.linspace(_min, _max, 15).tolist()[1:-1]
    # recreate p from x range
    p_range = [dist.cdf(i) for i in x_range]

    # expand p and x with more points
    _ps: List[float] = [dist.cdf(_min), dist.cdf(_max)] + p_range + [-0.1, 1.1]
    _xs: List[HPSampledValue] = [_min, _max] + x_range + [_min, _max]

    # inverse with icdf to find x again and assert x_icdf == x
    for p, x in zip(_ps, _xs):
        x_icdf = dist.icdf(p)
        assert abs(x_icdf - x) <= tolerance, f"`x_icdf != x` : `{x_icdf} != {x}`, given `p={p}`"
