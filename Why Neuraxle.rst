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
currently already writing savers for TensorFlow in the
`Neuraxle-TensorFlow <https://github.com/Neuraxio/Neuraxle-TensorFlow>`__
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
project <https://www.neuraxio.com/pages/neuraxios-time-series-solution>`__, which is
a premium software package built on top of Neuraxle for business boost
their time series machine learning projects by providing out-of-the-box
specialized pipeline steps. Some of those specialized steps are featured
in the `Deep Learning Pipelines <#deep-learning-pipelines>`__ section above.

Note: `the Neuraxio Time Series project <https://www.neuraxio.com/pages/neuraxios-time-series-solution>`__ is different from the Neuraxle project; those are separate projects. Neuraxio is commited to build open-source software, and defines itself as an open-source company. Learn more on `Neuraxle's license <#license>`__. The Neuraxle library is free and will always stay free, while Neuraxio Time Series is a premium add-on to Neuraxle.

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

