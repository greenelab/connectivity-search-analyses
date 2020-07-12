# Hetnet connectivity search prototyping and data repository

[![Build Status](https://travis-ci.org/greenelab/hetmech.svg?branch=master)](https://travis-ci.org/greenelab/hetmech)

Hetmech is a project to extract mechanistic connections between nodes in hetnets.
The project aims to identify the relevant network connections between a set of query nodes.
The method is designed to operate on hetnets (networks with multiple node or relationship types).

**Note: the `hetmech` python package has been renamed to `hetmatpy` and relocated to [`hetio/hetmatpy`](https://github.com/hetio/hetmatpy).**
This repository is now used as a historical archive, as well as a dataset storage and method prototyping repository.

Many findings from this repository are described in the [Connectivity Search Manuscript](https://greenelab.github.io/connectivity-search-manuscript/ "Hetnet connectivity search provides rapid insights into how two biomedical entities are related").
The manuscript source code is available in [`greenelab/connectivity-search-manuscript`](https://github.com/greenelab/connectivity-search-manuscript).

## Environment

This repository uses [conda](http://conda.pydata.org/docs/) to manage its environment as specified in [`environment.yml`](environment.yml).
Install the environment with:

```sh
# install new hetmech environment
conda env create --file=environment.yml

# update existing hetmech environment
conda env update --file=environment.yml
```

Then use `conda activate hetmech` and `conda deactivate` to activate or deactivate the environment.

## Acknowledgments

This work was supported through a research collaboration with [Pfizer Worldwide Research and Development](https://www.pfizer.com/partners/research-and-development).
This work was funded in part by the Gordon and Betty Moore Foundation’s Data-Driven Discovery Initiative through Grants [GBMF4552](https://www.moore.org/grant-detail?grantId=GBMF4552) to Casey Greene and [GBMF4560](https://www.moore.org/grant-detail?grantId=GBMF4560) to Blair Sullivan.
