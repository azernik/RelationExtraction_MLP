from setuptools import setup, find_packages

setup(
    name="relation_extraction",
    version="1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch",
        "numpy",
        "pandas",
        "scikit-learn",
        "nltk",
        "gensim",
        "pyyaml"
    ],
)
