Neuraxle Pipelines
==================

    Code Machine Learning Pipelines - The Right Way.

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
    y_test_predicted = p.transform(X_test)

    # Easy REST API deployment.
    app = FlaskRestApiWrapper(
        json_decoder=CustomJSONDecoderFor2DArray(),
        wrapped=p,
        json_encoder=CustomJSONEncoderOfOutputs(),
    ).get_app()
    app.run(debug=False, port=5000)

Visit the
`examples <https://www.neuraxle.org/stable/examples/index.html>`__
to get more a feeling of how it works, and inspiration.

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

Parallel Computing
~~~~~~~~~~~~~~~~~~

Neuraxle offer multiple parallel processing features using
`joblib <https://joblib.readthedocs.io/en/latest/parallel.html>`__. Most
parallel processing in Neuraxle happens in the
`pipeline <https://www.neuraxle.org/stable/api/neuraxle.pipeline.html>`__
and
`union <https://www.neuraxle.org/stable/api/neuraxle.union.html>`__
modules, and as such, neuraxle can be easily parallelized on a cluster
of computers using `distributed <https://ml.dask.org/joblib.html>`__ as
its `joblib backend <https://ml.dask.org/joblib.html>`__.

Time Series Processing
~~~~~~~~~~~~~~~~~~~~~~

Although Neuraxle is not limited to just time series processing
projects, it's especially good for those projects, as one of the goals
of Neuraxle is to provides a few abstractions that are useful for time
series projects.

With the various abstractions that Neuraxle provides, it's easy to get
started building a time-series processing project. Neuraxle is also the
backbone of `the Neuraxio Time Series
project <https://www.neuraxio.com/en/time-series-solution>`__, which is
a premium software package built on top of Neuraxle for business boost
their time series machine learning projects by providing out-of-the-box
specialized pipeline steps.

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
