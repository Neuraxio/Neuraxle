

class Trial_BaseAggregate:

    def __init__(self, context: AutoMLContext, trial: 'TrialManager', continue_loop_on_error: bool):
        super().__init__(context)



    def __enter__(self) -> 'Trial':
        self.flow.log_start()
        raise NotImplementedError("TODO: ???")

        self.trial.status = TrialStatus.RUNNING
        self.logger.info(
            '\nnew trial: {}'.format(
                json.dumps(self.hyperparams.to_nested_dict(), sort_keys=True, indent=4)))
        self.save_trial()
        return self

    def new_trial_split(self) -> 'TrialSplit':
        trial_split: TrialSplitManager = self.trial.new_validation_split()
        self.repo.set(self.loc, trial_split, False)
        self.repo.set(self.loc.with_dc(trial_split), trial_split, True)
        return TrialSplit(self.context, self.repo, trial_split)

    def __exit__(self, exc_type: Type, exc_val: Exception, exc_tb: traceback):
        gc.collect()
        raise NotImplementedError("TODO: log split result with MetricResultMetadata?")

        try:
            if exc_type is None:
                self.trial.end(status=TrialStatus.SUCCESS)
            else:
                # TODO: if ctrl+c, raise KeyboardInterrupt? log aborted or log failed?
                raise exc_val
        finally:
            self.save_trial()
            self._free_logger_file()


class TrialSplit_BaseAggregate:
    def __init__(self, context: AutoMLContext, repo: HyperparamsRepository, n_epochs: int):
        super().__init__(context, repo)
        self.n_epochs = n_epochs

    def __enter__(self):
        """
        Start trial, and set the trial status to PLANNED.
        """
        # TODO: ??
        self.trial_split.status = TrialStatus.RUNNING
        self.save_parent_trial()
        return self

    def new_epoch(self, epoch: int) -> 'Epoch':
        return Epoch(self.context, self.repo, epoch, self.n_epochs)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Stop trial, and save end time.

        :param exc_type:
        :param exc_val:
        :param exc_tb:
        :return:
        """
        self.end_time = datetime.datetime.now()
        if self.delete_pipeline_on_completion:
            del self.pipeline
        if exc_type is not None:
            self.set_failed(exc_val)
            self.trial_split.end(TrialStatus.FAILED)
            self.save_parent_trial()
            return False

        self.trial_split.end(TrialStatus.SUCCESS)
        self.save_parent_trial()
        return True

    def __exit__(self, exc_type: Type, exc_val: Exception, exc_tb: traceback):
        raise NotImplementedError("TODO: log split result with MetricResultMetadata?")

        if exc_val is None:
            self.flow.log_success()
        elif exc_type in self.error_types_to_raise:
            self.flow.log_failure(exc_val)
            return False  # re-raise.
        else:
            self.flow.log_error(exc_val)
            self.flow.log_aborted()
            return True  # don't re-raise.


class Epoch_BaseAggregate:
    def __init__(self, context: AutoMLContext, repo: HyperparamsRepository, epoch: int, n_epochs: int):
        super().__init__(context, repo)
        self.epoch: int = epoch
        self.n_epochs: int = n_epochs

    def __enter__(self) -> 'Epoch':
        self.flow.log_epoch(self.epoch, self.n_epochs)
        return self

    def __exit__(self, exc_type: Type, exc_val: Exception, exc_tb: traceback):
        self.flow.log_error(exc_val)
        raise NotImplementedError("TODO: log MetricResultMetadata?")
