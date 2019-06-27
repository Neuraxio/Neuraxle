from setuptools import setup, find_packages

with open('README.rst') as _f:
    _README = _f.read()

_VERSION = '0.1.0'

setup(
    name='neuraxle',
    version=_VERSION,
    description='Neuraxle is a Machine Learning (ML) library for building neat pipelines, providing the right '
                'abstractions to both ease research, development, and deployment of your ML applications.',
    long_description=_README,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Manufacturing",
        "Intended Audience :: Science/Research",
        "Intended Audience :: System Administrators",
        "Intended Audience :: Telecommunications Industry",
        'License :: OSI Approved :: Apache Software License',
        "Natural Language :: English",
        "Operating System :: OS Independent",
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        "Topic :: Adaptive Technologies",
        "Topic :: Office/Business",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Artificial Life",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Interface Engine/Protocol Translator",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development",
        "Topic :: Software Development :: Assemblers",
        "Topic :: Software Development :: Build Tools",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        # Topic :: Software Development :: Object Brokering,
        "Topic :: Software Development :: Pre-processors",
        "Topic :: Software Development :: Quality Assurance",
        "Topic :: Software Development :: Testing",
        "Topic :: System",
        # Topic :: System :: Clustering,
        # Topic :: System :: Distributed Computing,
        # Topic :: System :: Networking,
        # Topic :: System :: Systems Administration,
        "Topic :: Text Processing",
        "Topic :: Text Processing :: Filters",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Utilities",
        "Typing :: Typed"
    ],
    url='https://github.com/Neuraxio/Neuraxle',
    download_url='https://github.com/Neuraxio/Neuraxle/tarball/{}'.format(_VERSION),
    author='Neuraxio Inc.',
    author_email='guillaume.chevalier@neuraxio.com',
    packages=find_packages(include=['neuraxle*']),
    test_suite="testing",
    setup_requires=["pytest-runner"],
    install_requires=['numpy', 'scipy', 'matplotlib', 'scikit-learn>=0.20.3', 'keras',
                      'joblib>=0.13.2'],  # , 'tensorflow'],
    tests_require=["pytest", "pytest-cov"],
    include_package_data=True,
    license='Apache 2.0',
    keywords='pipeline pipelines data science machine learning deep learning'
)

print("")
print("--- If tensorflow isn't already installed, you must install it manually if you plan on using it in Neuraxle.")
print("")
