from setuptools import setup, find_packages

setup(
    name="IndicTransToolkit",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "sentencepiece",
        "indic-nlp-library-IT2 @ git+https://github.com/VarunGumma/indic_nlp_library.git",
    ],
    python_requires=">=3.8",
)
