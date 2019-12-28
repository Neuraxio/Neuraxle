
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

.. image:: https://img.shields.io/github/commit-activity/m/neuraxio/neuraxle?   :alt: GitHub commit activity
    :target: https://github.com/Neuraxio/Neuraxle

.. image:: https://img.shields.io/github/v/release/neuraxio/neuraxle?   :alt: GitHub release (latest by date)
    :target: https://pypi.org/project/neuraxle/


.. image:: https://www.neuraxio.com/en/blog/assets/pipeline_1_small.jpg

Neuraxle is a Machine Learning (ML) library for building neat pipelines,
providing the right abstractions to both ease research, development, and
deployment of your ML applications.

Installation
------------

Simply do:

.. code:: bash

    pip install neuraxle


Quickstart
----------

One of Neuraxle's most important goals is to make powerful machine
learning pipelines easy to build and deploy. Using Neuraxle should be
light, painless and obvious, yet without sacrificing powerfulness,
performance, nor possibilities.

For example, you can build a pipeline composed of multiple steps as
such:

.. code:: python

    p = Pipeline([
        # A Pipeline is composed of multiple chained steps. Steps
        # can alter the data before passing it to the next steps.
        AddFeatures([
            # Add (concatenate) features in parallel, that are
            # themselves derived of the existing features:
            PCA(n_components=2),
            FastICA(n_components=2),
        ]),
        RidgeModelStacking([
            # Here is an ensemble of 4 models or feature extractors,
            # That are themselves then fed to a ridge regression which
            # will act as a judge to finalize the prediction.
            LinearRegression(),
            LogisticRegression(),
            GradientBoostingRegressor(n_estimators=500),
            GradientBoostingRegressor(max_depth=5),
            KMeans(),
        ])
    ])
    # Note: here all the steps were imported from scikit-learn,
    # but the goal is that you can also define your own as needed.
    # Also note that a pipeline is a step itself: you can nest them.

    # The pipeline will learn on the data and acquire state.
    p = p.fit(X_train, y_train)

    # Once it learned, the pipeline can process new and
    # unseen data for making predictions.
    y_test_predicted = p.predict(X_test)

Visit the
`examples <https://www.neuraxle.org/stable/examples/index.html>`__
to get more a feeling of how it works, and inspiration.

Deep Learning Pipelines
-----------------------

Here is how to use deep learning algorithms within a Neuraxle Pipeline.

Deep Learning Pipeline Definition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Defining a Deep Learning pipeline is more complex. 
It needs a composition of many steps to: 

-  Loop on data for many epochs, but just during training.
-  Shuffle the data, just during training.
-  Use minibatches to process the data, which avoids to blow RAM. Your steps will fit incrementally.
-  Process data that is 2D, 3D, 4D, or even 5D or ND, with transformers made for 2D data slices.
-  Actually use your Deep Learning algorithm within your pipeline for it to learn and predict.

Below, we define a pipeline for time series classification using
a LSTM RNN. It includes data preprocessing steps as well as the
data flow management. `Time Series data is 3D <https://qr.ae/TZjoMb>`__.

.. code:: python
    
    deep_learning_seq_classif_pipeline = EpochRepeater(Pipeline([
        # X data shape: (batch, different_lengths, n_feature_columns)
        # y data shape: (batch, different_lengths)
        # Split X and Y into windows using 
        # an InputAndOutputTransformerMixin
        # abstraction to transform y too:
        SliceTimeSeries(window_size=128, last_label_as_seq_label=True),
        # X data shape: (more_than_batch, 128, n_feature_columns)
        # y data shape: (more_than_batch, 128)
        TrainOnlyWrapper(DataShuffler(seed=42)),
        MiniBatchSequentialPipeline([
            # X data shape: (batch_size, 128, n_feature_columns)
            # y data shape: (batch_size, 128)
            # Loop on 2D slices of the batch's 3D time series
            # data cube to apply 2D transformers:
            ForEachDataInput(Pipeline([
                # X data shape: (128, n_feature_columns)
                # y data shape: (128)
                # This step will load the lazy-loadable data
                # into a brick:
                ToNumpy(np_dtype=np.float32),
                # Fill nan and inf values with 0:
                DefaultValuesFiller(0.0),
                # Transform the columns (that is the innermost
                # axis/dim of data named `n_feature_columns`):
                ColumnTransformer([
                    # Columns 0, 1, 2, 3 and 4 needs to be
                    # normalized by mean and variance (std):
                    (range(0, 5), MeanVarianceNormalizer()),
                    # Column 5 needs to have it's `log plus 1` 
                    # value taken before normalization.
                    (5, Pipeline([
                        Log1P(), 
                        MeanVarianceNormalizer()
                    ]))
                    # Note that omited columns are discarded. 
                    # Also, multiple transformers on a column will 
                    # concatenate the results. 
                ]),
                # Transform the labels' indices to one-hot vectors.
                OutputTransformerWrapper(
                    OneHotEncoder(nb_columns=6, name='labels'))
                # X data shape: (128, n_feature_columns)
                # y data shape: (128, 6)
            ])),
            # X data shape: (batch_size, 128, n_feature_columns)
            # y data shape: (batch_size, 128, 6)
            # Classification with a deep neural network,
            # using the Neuraxle-TensorFlow and/or
            # Neuraxle-PyTorch extensions:
            ClassificationLSTM(n_stacked=2, n_residual=3),
            # X data shape: (batch_size, 128, 6)
            # y data shape: (batch_size, 128, 6)
        ], batch_size=32),
        # X data shape: (batch_size, 128, 6)
    ]), epochs=200, fit_only=True)

Deep Learning Pipeline Training and Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here we train and evaluate with a train-validation split. Note that
automatic hyperparameter tuning would require only a few more lines
of code: see our
`hyperparameter tuning example <https://www.neuraxle.org/stable/examples/boston_housing_meta_optimization.html#sphx-glr-examples-boston-housing-meta-optimization-py>`__.

.. code:: python

    # Wrap the pipeline by a validation strategy,
    # this could have been Cross Validation as well:
    training_pipeline = ValidationSplitWrapper(
        deep_learning_seq_classif_pipeline,
        val_size=0.1,
        scoring_function=sklearn.metrics.accuracy_score
    )

    # Fitting and evaluating the pipeline.
    # X_train data shape: (batch, different_lengths, n_feature_columns)
    # y_train data shape: (batch, different_lengths)
    training_pipeline.fit(X_train, y_train)
    # Note that X_train and y_train can be lazy loaders.
    print('Train accuracy: {}'.format(
        training_pipeline.scores_train_mean))
    print('Validation accuracy: {}'.format(
        training_pipeline.scores_validation_mean))

    # Recover the pipeline in test mode:
    production_pipeline = training_pipeline.get_step()
    production_pipeline.set_train(False)

Deep Learning Production Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Deploying your deep learning app to a JSON REST API. Refer
to `Flask's deployment documentation <https://flask.palletsprojects.com/en/1.1.x/tutorial/deploy/>`__
for more info on deployment servers and security.

.. code:: python

    # Will now serve the pipeline to a REST API as an example:
    # Note that having saved the pipeline to disk
    # (for reloading this in another file) would be easy, too, using savers.
    app = FlaskRestApiWrapper(
        json_decoder=YourCustomJSONDecoderFor2DArray(),
        wrapped=production_pipeline,
        json_encoder=YourCustomJSONEncoderOfOutputs()
    ).get_app()
    app.run(debug=False, port=5000)

Calling a Deployed Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This could be ran from another distant computer to call your app:

.. code:: python

    p = APICaller(
        json_encoder=YourCustomJSONEncoderOfInputs(),
        url="http://127.0.0.1:5000/",
        json_decoder=YourCustomJSONDecoderOfOutputs()
    )
    y_pred = p.predict(X_test)
    print(y_pred)

Note that we'll soon have better remote proxy design patterns for distant
pipelines, and distant parallel processing and distant parallel training.

Why Neuraxle?
-------------

Production-ready
~~~~~~~~~~~~~~~~

Most research projects don't ever get to production. However, you want
your project to be production-ready and already adaptable (clean) by the
time you finish it. You also want things to be simple so that you can
get started quickly.

Most existing machine learning pipeline frameworks are either too simple
or too complicated for medium-scale projects. Neuraxle is balanced for
medium-scale projects, providing simple, yet powerful abstractions that
are ready to be used.

Compatibility
~~~~~~~~~~~~~

    Neuraxle is built as a framework that enables you to define your own
    pipeline steps.

This means that you can use
`scikit-learn <https://arxiv.org/pdf/1201.0490v4.pdf>`__,
`Keras <https://keras.io/>`__,
`TensorFlow <https://arxiv.org/pdf/1603.04467v2.pdf>`__,
`PyTorch <https://openreview.net/pdf?id=BJJsrmfCZ>`__,
`Hyperopt <https://pdfs.semanticscholar.org/d4f4/9717c9adb46137f49606ebbdf17e3598b5a5.pdf>`__,
`Ray <https://arxiv.org/pdf/1712.05889v2.pdf>`__ 
and/or **any other machine learning library** you like within and
throughout your Neuraxle pipelines.

Parallel Computing and Serialization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Neuraxle offer multiple parallel processing features. One magical thing 
that we did are Savers. Savers allow you to define how a step can be 
serialized. This way, it's possible to avoid Python's parallel 
processing limitations and pitfalls. 

Let's suppose that your pipeline has a step that imports code from
another library and that this code isn't serializable (e.g.: some
code written in C++ and interacting with the GPUs or anything funky).
To make this step serializable, just define a saver which will tell
the step how to dump itself to disk and reload itself. This will
allow the step to be sent to a remote computer or to be threadable
by reloading the save. The save can be dumped to a RAM disk for
more performance and avoid truly writing to disks.

Neuraxle is compatible with most other ML and DL libraries. We're
currently already writing savers for PyTorch and TensorFlow in the
`Neuraxle-PyTorch <https://github.com/Neuraxio/Neuraxle-PyTorch>`__ 
and `Neuraxle-TensorFlow<https://github.com/Neuraxio/Neuraxle-TensorFlow>`__ 
extensions of this project.

Time Series Processing
~~~~~~~~~~~~~~~~~~~~~~

Although Neuraxle is not limited to just time series processing
projects, it's especially good for those projects, as one of the goals
of Neuraxle is to provides a few abstractions that are useful for time
series projects, as
`Time Series data is often 3D <https://qr.ae/TZjoMb>`__ or even ND.

With the various abstractions that Neuraxle provides, it's easy to get
started building a time-series processing project. Neuraxle is also the
backbone of `the Neuraxio Time Series
project <https://www.neuraxio.com/en/time-series-solution>`__, which is
a premium software package built on top of Neuraxle for business boost
their time series machine learning projects by providing out-of-the-box
specialized pipeline steps. Some of those specialized steps are featured
in the `Deep Learning Pipelines <#deep-learning-pipelines>`__ section above.

Note: `the Neuraxio Time Series
project <https://www.neuraxio.com/en/time-series-solution>`__ is
different from the Neuraxle project, those are separate projects.
Neuraxio is commited to build open-source software, and defines itself
as an open-source company. Learn more on `Neuraxle's
license <#license>`__. The Neuraxle library is free and will always stay
free, while Neuraxio Time Series is a premium add-on to Neuraxle.

Automatic Machine Learning
~~~~~~~~~~~~~~~~~~~~~~~~~~

One of the core goal of this framework is to enable easy automatic
machine learning, and also meta-learning. It should be easy to train a
meta-optimizer on many different tasks: the optimizer is a model itself
that maps features of datasets and features of the hyperparameter space
to a guessed performance score to predict the best hyperparameters.
Hyperparameter spaces are easily defined with a range, and are only
coupled to their respective pipeline steps, rather than being coupled to
the whole pipeline, which enable class reuse and more modularity.

Comparison to Other Machine Learning Pipeline Frameworks
--------------------------------------------------------

scikit-learn
~~~~~~~~~~~~

Everything that works in sklearn is also useable in Neuraxle. Neuraxle
is built in a way that does not replace what already exists. Therefore,
Neuraxle adds more power to scikit-lean by providing neat abstractions,
and neuraxle is even retrocompatible with sklean if it ever needed to be
included in an already-existing sklearn pipeline (you can do that by
using ``.tosklearn()`` on your Neuraxle pipeline). We believe that
Neuraxle helps scikit-learn, and also scikit-learn will help Neuraxle.
Neuraxle is best used with scikit-learn.

Also, the top core developers of scikit-learn, Andreas C. Müller, `gave
a talk <https://www.youtube.com/embed/Wy6EKjJT79M>`__ in which he lists
the elements that are yet to be done in scikit-learn. He refers to
building bigger pipelines with automatic machine learning, meta
learning, improving the abstractions of the search spaces, and he also
points out that it would be possible do achieve that in another library
which could reuse scikit-learn. Neuraxle is here to solve those problems
that are actually shared by the open-source community in general. Let's
move forward with Neuraxle: join Neuraxle's `community <#community>`__.

.. raw:: html

   <iframe width="560" height="315" src="https://www.youtube.com/embed/Wy6EKjJT79M?start=1361&amp;end=1528" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen>

.. raw:: html

   </iframe>

Apache Beam
~~~~~~~~~~~

Apache Beam is a big, multi-language project and hence is complicated.
Neuraxle is pythonic and user-friendly: it's easy to get started.

Also, it seems that Apache Beam has GPL and MPL dependencies, which
means Apache Beam might itself be copyleft (?). Neuraxle doesn't have
such copyleft dependencies.

spaCy
~~~~~

spaCy has copyleft dependencies or may download copyleft content, and it
is built only for Natural Language Processing (NLP) projects. Neuraxle
is open to any kind of machine learning projects and isn't an NLP-first
project.

Kubeflow
~~~~~~~~

Kubeflow is cloud-first, using Kubernetes and is more oriented towards
devops. Neuraxle isn't built as a cloud-first solution and isn't tied to
Kubernetes. Neuraxle instead offers many parallel processing features,
such as the ability to be scaled on many cores of a computer, and even
on a computer cluster (e.g.: in the cloud using any cloud provider) with
joblib, using dask's distributed library as a joblib backend. A Neuraxle
project is best deployed as a microservice within your software
environment, and you can fully control and customize how you deploy your
project (e.g.: coding yourself a pipeline step that does json conversion
to accept http requests).


Community
---------

Join our `Slack
workspace <https://neuraxio-open-source.slack.com/join/shared_invite/enQtNjc0NzM1NTI5MTczLWUwZmI5NjhkMzRmYzc1MGE5ZTE0YWRkYWI3NWIzZjc1YTRlM2Y1MzRmYzFmM2FiNWNhNGZlZDhhMzkyMTQ1ZTQ>`__ and our `Gitter <https://gitter.im/Neuraxle/community>`__!
We <3 collaborators. You can also subscribe to our `mailing list <https://www.neuraxio.com/en/blog/index.html>`__ where we post our updates and news. 

For **technical questions**, we recommend posting them on
`StackOverflow <https://stackoverflow.com/questions/tagged/machine-learning>`__
first with ``neuraxle`` in the tags (amongst probably ``python`` and
``machine-learning``), and *then* opening an
`issue <https://github.com/Neuraxio/Neuraxle/issues>`__ to link to your
Stack Overflow question.

For **suggestions, comments, and issues**, don't hesitate to open an
`issue <https://github.com/Neuraxio/Neuraxle/issues>`__.

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

License
~~~~~~~

Neuraxle is licensed under the `Apache License, Version
2.0 <https://github.com/Neuraxio/Neuraxle/blob/master/LICENSE>`__.

Summary of the License
^^^^^^^^^^^^^^^^^^^^^^

At `Neuraxio <https://www.neuraxio.com/en/>`__, we have open-source at
heart. We want *you* to be able to use Neuraxio's Neuraxle as much as
possible without copyleft restrictions. For this reasons, Neuraxle don't
depend on copyleft librairies and is neither licensed under a copyleft
license. This way, Neuraxle is quite permissive.

The License is very permissive and not very restrictive.

Permissions:
 - Commercial use
 - Modification
 - Distribution
 - Patent use
 - Private use

Limitations:
 - Trademark use
 - Liability
 - Warranty

Conditions:
 - License and copyright notice
 - State changes

For example, if Neuraxle is used within a larger project, it doesn't
necessarily mean that the larger project is also licensed under the same
license. Licensed works, modifications, and larger works may be
distributed under different terms and without source code.

Note: this Summary of the License is not legal advice. Refer to the `full
license <https://github.com/Neuraxio/Neuraxle/blob/master/LICENSE>`__.

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

Supported By
~~~~~~~~~~~~

We thank these organisations for generously supporting the project:

-  Neuraxio Inc.: https://github.com/Neuraxio


.. raw:: html

    <img src="https://www.neuraxio.com/images/neuraxio_logo_transparent.png" width="140px">


-  Umanéo Technologies Inc.: https://www.umaneo.com/

.. raw:: html

    <img src="https://uploads-ssl.webflow.com/5be35e61c9728278fc5f4150/5c6dabf76fc786262e6654a0_signature-courriel-logo-umaneo.png" width="200px">


-  Solution Nexam Inc.: https://www.nexam.io/

.. raw:: html

    <img src="https://www.neuraxio.com/images/solution_nexam_io.jpg" width="180px">


-  La Cité, LP: http://www.lacitelp.com/

.. raw:: html

    <img src="https://www.neuraxio.com/images/La-Cite-LP.png" width="260">

Support Us
~~~~~~~~~~~~

-  `Get in touch with us <https://gitter.im/Neuraxle/community>`__.
-  `Be a sponsor <https://www.neuraxio.com/en/>`__.
-  Save this for later:

.. image:: https://img.shields.io/github/watchers/Neuraxio/Neuraxle?style=social&   :alt: GitHub watchers
    :target: https://github.com/Neuraxio/Neuraxle/watchers
