from neuraxle.base import BaseStep, TruncableSteps, NonFittableMixin, NamedTupleList
from neuraxle.steps.numpy import NumpyConcatenateInnerFeatures


class ModelStacking(TruncableSteps):
    def __init__(self, brothers: NamedTupleList, judge: BaseStep, joiner: BaseStep):
        super().__init__(brothers)
        self.joiner: BaseStep = joiner
        self.judge: BaseStep = judge

    def fit(self, data_inputs, expected_outputs=None):
        # TODO: FeatureUnion?
        results = []
        for _, bro in self.steps_as_tuple:
            res = bro.fit_transform(data_inputs, expected_outputs)
            results.append(res)

        results = self.joiner.fit_transform(results)

        self.judge.fit(results, expected_outputs)
        return self

    def transform(self, data_inputs):
        results = []
        for name, bro in self.steps_as_tuple:
            res = bro.transform(data_inputs)
            results.append(res)

        results = self.joiner.transform(results)

        return self.judge.transform(results)


class FeatureUnion(TruncableSteps):
    # TODO: parallel.
    def __init__(self, steps_as_tuple: NamedTupleList, joiner: NonFittableMixin = NumpyConcatenateInnerFeatures()):
        super().__init__(steps_as_tuple)
        self.joiner = joiner  # TODO: add "other" types of step(s) to TuncableSteps or to another intermediate class.

    def fit(self, data_inputs, expected_outputs=None):
        for _, bro in self.steps_as_tuple:
            bro.fit(data_inputs, expected_outputs)
        return self

    def transform(self, data_inputs):
        results = []
        for _, bro in self.steps_as_tuple:
            res = bro.transform(data_inputs)
            results.append(res)
        results = self.joiner.transform(results)
        return results


class Identity(NonFittableMixin, BaseStep):
    """A pipeline step that has no effect.

    This can be useful to concatenate new features to existing features, such as what AddFeatures do.
    """

    def __init__(self):
        """
        Create an Identity BaseStep that is also a NonFittableMixin (doesn't require fitting).
        """
        BaseStep.__init__(self)

    def transform(self, data_inputs):
        """
        Do nothing - return the same data.

        :param data_inputs:
        :return: the `data_inputs`, unchanged.
        """
        return data_inputs

    def transform_one(self, data_input):
        """
        Do nothing - return the same data.

        :param data_input:
        :return: the `data_input`, unchanged.
        """
        return data_input

    def inverse_transform(self, processed_outputs):
        """
        Do nothing - return the same data.

        :param processed_outputs:
        :return: the `processed_outputs`, unchanged.
        """
        return processed_outputs

    def inverse_transform_one(self, data_output):
        """
        Do nothing - return the same data.

        :param data_output:
        :return: the `data_output`, unchanged.
        """
        return data_output


class AddFeatures(FeatureUnion):
    def __init__(self, steps_as_tuple: NamedTupleList, **kwargs):
        super().__init__([Identity()] + steps_as_tuple, **kwargs)
