
Neuraxle Pipelines
==================

Code Machine Learning Pipelines - The Right Way.

.. image:: https://img.shields.io/github/workflow/status/Neuraxio/Neuraxle/Test%20Python%20Package/master?   :alt: Build
    :target: https://github.com/Neuraxio/Neuraxle

.. image:: https://img.shields.io/gitter/room/Neuraxio/Neuraxle?   :alt: Gitter
    :target: https://gitter.im/Neuraxle/community

.. image:: https://img.shields.io/pypi/l/neuraxle?   :alt: PyPI - License
    :target: https://www.neuraxle.org/stable/Neuraxle/README.html#license

.. image:: https://img.shields.io/pypi/dm/neuraxle?   :alt: PyPI - Downloads
    :target: https://pypi.org/project/neuraxle/

.. image:: https://img.shields.io/github/v/release/neuraxio/neuraxle?   :alt: GitHub release (latest by date)
    :target: https://pypi.org/project/neuraxle/

.. image:: https://img.shields.io/codacy/grade/d56d39746e29468bac700ee055694192?   :alt: Codacy
    :target: https://www.codacy.com/gh/Neuraxio/Neuraxle/dashboard

.. image:: assets/images/neuraxle_logo.png
    :alt: Neuraxle Logo
    :align: center

Neuraxle is a Machine Learning (ML) library for building machine learning pipelines.

- **Component-Based**: Build encapsulated steps, then compose them to build complex pipelines.
- **Evolving State**: Each pipeline step can fit, and evolve through the learning process
- **Hyperparameter Tuning**: Optimize your pipelines using AutoML, where each pipeline step has their own hyperparameter space.
- **Compatible**: Use your favorite machine learning libraries inside and outside Neuraxle pipelines.
- **Production Ready**: Pipeline steps can manage how they are saved by themselves, and the lifecycle of the objects allow for train, and test modes.
- **Streaming Pipeline**: Transform data in many pipeline steps at the same time in parallel using multiprocessing Queues.

Documentation
-------------

You can find the Neuraxle documentation `on the website <https://www.neuraxle.org/stable/index.html>`_. It also contains multiple examples demonstrating some of its features.


Installation
------------

Simply do:

.. code:: bash

    pip install neuraxle


Examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We have several `examples on the website <https://www.neuraxle.org/stable/examples/index.html>`__.

For example, you can build a time series processing pipeline as such:

.. code:: python

    p = Pipeline([
        TrainOnly(DataShuffler()),
        WindowTimeSeries(),
        MiniBatchSequentialPipeline([
            Tensorflow2ModelStep(
                create_model=create_model,
                create_optimizer=create_optimizer,
                create_loss=create_loss
            ).set_hyperparams(HyperparameterSpace({
                'hidden_dim': 12,
                'layers_stacked_count': 2,
                'lambda_loss_amount': 0.0003,
                'learning_rate': 0.001
                'window_size_future': sequence_length,
                'output_dim': output_dim,
                'input_dim': input_dim
            })).set_hyperparams_space(HyperparameterSpace({
                'hidden_dim': RandInt(6, 750),
                'layers_stacked_count': RandInt(1, 4),
                'lambda_loss_amount': Uniform(0.0003, 0.001),
                'learning_rate': Uniform(0.001, 0.01),
                'window_size_future': FixedHyperparameter(sequence_length),
                'output_dim': FixedHyperparameter(output_dim),
                'input_dim': FixedHyperparameter(input_dim)
            }))
        ])
    ])

    # Load data
    X_train, y_train, X_test, y_test = generate_classification_data()

    # The pipeline will learn on the data and acquire state.
    p = p.fit(X_train, y_train)

    # Once it learned, the pipeline can process new and
    # unseen data for making predictions.
    y_test_predicted = p.predict(X_test)

You can also tune your hyperparameters using AutoML algorithms such as the TPE:

.. code:: python

    auto_ml = AutoML(
        pipeline=pipeline,
        hyperparams_optimizer=TreeParzenEstimatorHyperparameterSelectionStrategy(
            number_of_initial_random_step=10,
            quantile_threshold=0.3,
            number_good_trials_max_cap=25,
            number_possible_hyperparams_candidates=100,
            prior_weight=0.,
            use_linear_forgetting_weights=False,
            number_recent_trial_at_full_weights=25
        ),
        validation_splitter=ValidationSplitter(test_size=0.20),
        scoring_callback=ScoringCallback(accuracy_score, higher_score_is_better=True),
        callbacks[
            MetricCallback(f1_score, higher_score_is_better=True),
            MetricCallback(precision, higher_score_is_better=True),
            MetricCallback(recall, higher_score_is_better=True)
        ],
        n_trials=7,
        epochs=10,
        hyperparams_repository=HyperparamsJSONRepository(cache_folder='cache'),
        refit_trial=True,
    )

    # Load data, and launch AutoML loop !
    X_train, y_train, X_test, y_test = generate_classification_data()
    auto_ml = auto_ml.fit(X_train, y_train)

    # Get the model from the best trial, and make predictions using predict.
    best_pipeline = auto_ml.get_best_model()
    y_pred = best_pipeline.predict(X_test)


--------------
Why Neuraxle ?
--------------

Most research projects don't ever get to production. However, you want
your project to be production-ready and already adaptable (clean) by the
time you finish it. You also want things to be simple so that you can
get started quickly. Read more about `the why of Neuraxle here. <https://github.com/Neuraxio/Neuraxle/blob/master/Why%20Neuraxle.rst>`_

---------
Community
---------

For **technical questions**, please post them on
`StackOverflow <https://stackoverflow.com/questions/tagged/neuraxle>`__
using the ``neuraxle`` tag. The StackOverflow question will automatically
be posted in `Neuraxio's Slack
workspace <https://join.slack.com/t/neuraxio/shared_invite/zt-8lyw42c5-4PuWjTT8dQqeFK3at1s_dQ>`__ and our `Gitter <https://gitter.im/Neuraxle/community>`__ in the #Neuraxle channel. 

For **suggestions, feature requests, and error reports**, please
open an `issue <https://github.com/Neuraxio/Neuraxle/issues>`__.

For **contributors**, we recommend using the PyCharm code editor and to
let it manage the virtual environment, with the default code
auto-formatter, and using pytest as a test runner. To contribute, first
fork the project, then do your changes, and then open a pull request in
the main repository. Please make your pull request(s) editable, such as
for us to add you to the list of contributors if you didn't add the
entry, for example. Ensure that all tests run before opening a pull
request. You'll also agree that your contributions will be licensed
under the `Apache 2.0
License <https://github.com/Neuraxio/Neuraxle/blob/master/LICENSE>`__,
which is required for everyone to be able to use your open-source
contributions.

Finally, you can as well join our `Slack
workspace <https://join.slack.com/t/neuraxio/shared_invite/zt-8lyw42c5-4PuWjTT8dQqeFK3at1s_dQ>`__ and our `Gitter <https://gitter.im/Neuraxle/community>`__ to collaborate with us. We <3 collaborators. You can also subscribe to our mailing list where we will post some `updates and news <https://www.neuraxle.org/stable/intro.html>`__. 


License
~~~~~~~

Neuraxle is licensed under the `Apache License, Version
2.0 <https://github.com/Neuraxio/Neuraxle/blob/master/LICENSE>`__.

Citation
~~~~~~~~~~~~

You may cite our `extended abstract <https://www.researchgate.net/publication/337002011_Neuraxle_-_A_Python_Framework_for_Neat_Machine_Learning_Pipelines>`__ that was presented at the Montreal Artificial Intelligence Symposium (MAIS) 2019. Here is the bibtex code to cite:

.. code:: bibtex

    @misc{neuraxle,
    author = {Chevalier, Guillaume and Brillant, Alexandre and Hamel, Eric},
    year = {2019},
    month = {09},
    pages = {},
    title = {Neuraxle - A Python Framework for Neat Machine Learning Pipelines},
    doi = {10.13140/RG.2.2.33135.59043}
    }

Contributors
~~~~~~~~~~~~

Thanks to everyone who contributed to the project:

-  Guillaume Chevalier: https://github.com/guillaume-chevalier
-  Alexandre Brillant: https://github.com/alexbrillant
-  Éric Hamel: https://github.com/Eric2Hamel
-  Jérôme Blanchet: https://github.com/JeromeBlanchet
-  Michaël Lévesque-Dion: https://github.com/mlevesquedion
-  Philippe Racicot: https://github.com/Vaunorage
-  Neurodata: https://github.com/NeuroData-ltd
-  Klaimohelmi: https://github.com/Klaimohelmi
-  Vincent Antaki: https://github.com/vincent-antaki

Supported By
~~~~~~~~~~~~

We thank these organisations for generously supporting the project:

-  Neuraxio Inc.: https://github.com/Neuraxio

.. raw:: html

    <img src="https://raw.githubusercontent.com/Neuraxio/Neuraxle/master/assets/images/neuraxio.png" width="150px">

-  Umanéo Technologies Inc.: https://www.umaneo.com/

.. raw:: html

    <img src="https://raw.githubusercontent.com/Neuraxio/Neuraxle/master/assets/images/umaneo.png" width="200px">

-  Solution Nexam Inc.: https://nexam.io/

.. raw:: html

    <img src="https://raw.githubusercontent.com/Neuraxio/Neuraxle/master/assets/images/solution_nexam_io.jpg" width="180px">

-  La Cité, LP: https://www.lacitelp.com/accueil

.. raw:: html

    <img src="https://raw.githubusercontent.com/Neuraxio/Neuraxle/master/assets/images/La-Cite-LP.png" width="260px">

-  Kimoby: https://www.kimoby.com/

.. raw:: html

    <img src="https://raw.githubusercontent.com/Neuraxio/Neuraxle/master/assets/images/kimoby.png" width="200px">
