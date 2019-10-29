from neuraxle.base import MetaStepMixin, BaseStep, ExecutionContext, DataContainer, NonTransformableMixin, \
    ExecutionMode, NonFittableMixin, ForceHandleMixin


class TransformOnlyWrapper(
    NonTransformableMixin,
    NonFittableMixin,
    MetaStepMixin,
    BaseStep
):
    """
    A wrapper step that makes its wrapped step only executes in the transform execution mode.

    .. seealso:: :class:`ExecutionMode`,
        :class:`neuraxle.base.DataContainer`,
        :class:`neuraxle.base.NonTransformableMixin`,
        :class:`neuraxle.base.NonFittableMixin`,
        :class:`neuraxle.base.MetaStepMixin`,
        :class:`neuraxle.base.BaseStep`
    """

    def __init__(self, wrapped: BaseStep):
        NonTransformableMixin.__init__(self)
        NonFittableMixin.__init__(self)
        MetaStepMixin.__init__(self, wrapped=wrapped)
        BaseStep.__init__(self)

    def handle_transform(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        wrapped_context = context.push(self.wrapped)

        return self.wrapped.handle_transform(data_container, wrapped_context)


class FitTransformOnlyWrapper(
    NonTransformableMixin,
    NonFittableMixin,
    MetaStepMixin,
    BaseStep
):
    """
    A wrapper step that makes its wrapped step only executes in the fit_transform execution mode.

    .. seealso::
        :class:`neuraxle.base.ExecutionMode`,
        :class:`neuraxle.base.DataContainer`,
        :class:`neuraxle.base.NonTransformableMixin`,
        :class:`neuraxle.base.NonFittableMixin`,
        :class:`neuraxle.base.MetaStepMixin`,
        :class:`neuraxle.base.BaseStep`
    """

    def __init__(self, wrapped: BaseStep):
        NonTransformableMixin.__init__(self)
        NonFittableMixin.__init__(self)
        MetaStepMixin.__init__(self, wrapped=wrapped)
        BaseStep.__init__(self)

    def handle_fit_transform(self, data_container: DataContainer, context: ExecutionContext):
        wrapped_context = context.push(self.wrapped)

        self.wrapped, outputs = self.wrapped.handle_fit_transform(data_container, wrapped_context)

        return self, outputs


class FitOnlyWrapper(
    ForceHandleMixin,
    MetaStepMixin,
    BaseStep
):
    """
    A wrapper step that makes its wrapped step only executes in the fit execution mode.

    .. seealso::
        :class:`neuraxle.base.ExecutionMode`,
        :class:`neuraxle.base.DataContainer`,
        :class:`neuraxle.base.ForceHandleMixin`,
        :class:`neuraxle.base.MetaStepMixin`,
        :class:`neuraxle.base.BaseStep`
    """

    def __init__(self, wrapped: BaseStep):
        ForceHandleMixin.__init__(self)
        MetaStepMixin.__init__(self, wrapped=wrapped)
        BaseStep.__init__(self)

    def handle_transform(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        # fit only wrapper: nothing to do in transform
        return data_container

    def handle_fit_transform(self, data_container: DataContainer, context: ExecutionContext):
        if context.execution_mode == ExecutionMode.FIT:
            wrapped_context = context.push(self.wrapped)
            self.wrapped, data_container = self.wrapped.handle_fit_transform(data_container, wrapped_context)

        return self, data_container

    def handle_fit(self, data_container: DataContainer, context: ExecutionContext):
        self.wrapped, data_container = self.wrapped.handle_fit(data_container, context)
        return self, data_container
