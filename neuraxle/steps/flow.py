from neuraxle.base import MetaStepMixin, BaseStep, ExecutionContext, DataContainer, NonTransformableMixin, \
    ExecutionMode, NonFittableMixin


class TransformOnlyWrapper(NonTransformableMixin, NonFittableMixin, MetaStepMixin, BaseStep):
    """
    A wrapper step that only executes in the transform execution mode.

    .. seealso:: :class:`ExecutionMode`, :class:`DataContainer`, :class:`NonTransformableMixin`, :class:`NonFittableMixin`, :class:`MetaStepMixin`, :class:`BaseStep`
    """

    def handle_transform(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        return self.wrapped.handle_transform(data_container, context)


class FitTransformOnlyWrapper(NonTransformableMixin, NonFittableMixin, MetaStepMixin, BaseStep):
    """
    A wrapper step that only executes in the fit_transform execution mode.

    .. seealso:: :class:`ExecutionMode`, :class:`DataContainer`, :class:`NonTransformableMixin`, :class:`NonFittableMixin`, :class:`MetaStepMixin`, :class:`BaseStep`
    """

    def handle_fit_transform(self, data_container: DataContainer, context: ExecutionContext):
        self.wrapped, outputs = self.wrapped.handle_fit_transform(data_container, context)
        return self, outputs


class FitOnlyWrapper(NonTransformableMixin, NonFittableMixin, MetaStepMixin, BaseStep):
    """
    A wrapper step that only executes in the fit execution mode.

    .. seealso:: :class:`ExecutionMode`, :class:`DataContainer`, :class:`NonTransformableMixin`, :class:`NonFittableMixin`, :class:`MetaStepMixin`, :class:`BaseStep`
    """

    def handle_fit_transform(self, data_container: DataContainer, context: ExecutionContext):
        if context.execution_mode == ExecutionMode.FIT:
            self.wrapped, data_container = self.wrapped.handle_fit_transform(data_container, context)

        return self, data_container

    def handle_fit(self, data_container: DataContainer, context: ExecutionContext):
        self.wrapped, data_container = self.wrapped.handle_fit(data_container, context)
        return self, data_container
