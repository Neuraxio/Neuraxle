from neuraxle.base import MetaStepMixin, BaseStep, ExecutionContext, DataContainer, NonTransformableMixin, \
    ExecutionMode, NonFittableMixin, ResumableStepMixin


class ResumableMetaStepMixin(ResumableStepMixin):
    def should_resume(self, data_container: DataContainer, context: ExecutionContext) -> bool:
        if isinstance(self.wrapped, ResumableStepMixin):
            wrapped_context = context.push(self.wrapped)
            return self.wrapped.should_resume(data_container, wrapped_context)
        return False


class TransformOnlyWrapper(
    NonTransformableMixin,
    NonFittableMixin,
    MetaStepMixin,
    ResumableMetaStepMixin,
    BaseStep
):
    """
    A wrapper step that only executes in the transform execution mode.

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
        ResumableMetaStepMixin.__init__(self)
        BaseStep.__init__(self)

    def handle_transform(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        wrapped_context = context.push(self.wrapped)

        return self.wrapped.handle_transform(data_container, wrapped_context)


class FitTransformOnlyWrapper(
    NonTransformableMixin,
    NonFittableMixin,
    MetaStepMixin,
    ResumableMetaStepMixin,
    BaseStep
):
    """
    A wrapper step that only executes in the fit_transform execution mode.

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
        ResumableMetaStepMixin.__init__(self)
        BaseStep.__init__(self)

    def handle_fit_transform(self, data_container: DataContainer, context: ExecutionContext):
        wrapped_context = context.push(self.wrapped)

        self.wrapped, outputs = self.wrapped.handle_fit_transform(data_container, wrapped_context)

        return self, outputs


class FitOnlyWrapper(
    NonTransformableMixin,
    NonFittableMixin,
    MetaStepMixin,
    ResumableMetaStepMixin,
    BaseStep
):
    """
    A wrapper step that only executes in the fit execution mode.

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
        ResumableMetaStepMixin.__init__(self)
        BaseStep.__init__(self)

    def handle_fit_transform(self, data_container: DataContainer, context: ExecutionContext):
        if context.execution_mode == ExecutionMode.FIT:
            wrapped_context = context.push(self.wrapped)
            self.wrapped, data_container = self.wrapped.handle_fit_transform(data_container, wrapped_context)

        return self, data_container

    def handle_fit(self, data_container: DataContainer, context: ExecutionContext):
        self.wrapped, data_container = self.wrapped.handle_fit(data_container, context)
        return self, data_container
