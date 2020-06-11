import setuptools
from io import open

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cmdtools",
    version="0.0.2",
    author="Alexander Sikorski",
    author_email="sikorski@zib.de",
    description="A collection of tools relating to transfer operators",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://git.zib.de/cmd/cmdtools",
    packages=setuptools.find_packages('src'),
    package_dir={'': 'src'},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=['numpy', 'scipy'],
    extras_require={'slepc': ['petsc4py', 'slepc4py']}
)
