import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cmdtools",
    version="0.0.1",
    author="Alexander Sikorski",
    author_email="sikorski@zib.de",
    description="A collection of tools relating to transfer operators",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://git.zib.de/cmd/pccap",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=['numpy', 'scipy'],
    python_requires='>=3.6',
)