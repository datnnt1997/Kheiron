import pathlib
from kheiron import __version__
from setuptools import find_packages, setup

HERE = pathlib.Path(__file__).parent
README = (HERE / 'README.md').read_text()
requires = (HERE / 'requirements.txt').read_text().split('\n')

setup(
    name="Kheiron",
    version=__version__,
    description="Kheiron: Train and evaluate libary for Pytorch model.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/datnnt1997/Kheiron",
    author="DatNgo",
    author_email="datnnt97@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    packages=find_packages(exclude=('datasets', 'outputs')),
    include_package_data=True,
    install_requires=requires
)