# hetmech prototyping and data repository

[![Build Status](https://travis-ci.org/greenelab/hetmech.svg?branch=master)](https://travis-ci.org/greenelab/hetmech)

Hetmech is a project to extract mechanistic connections between nodes in hetnets.
The project aims to identify the relevant network connections between a set of query nodes.
The method is designed to operate on hetnets (networks with multiple node or relationship types).

**Note: the `hetmech` python package has been renamed to `hetmatpy` and relocated to [`hetio/hetmatpy`](https://github.com/hetio/hetmatpy).**
This repository is now used as a historical archive, as well as a dataset storage and method prototyping repository.
This project is still under development: use with caution.

## Environment

This repository uses [conda](http://conda.pydata.org/docs/) to manage its environment as specified in [`environment.yml`](environment.yml).
Install the environment with:

```sh
conda env create --file=environment.yml
```

Then use `conda activate hetmech` and `conda deactivate` to activate or deactivate the environment.

## Acknowledgments

This work is supported through a research collaboration with [Pfizer Worldwide Research and Development](https://www.pfizer.com/partners/research-and-development).
This work is funded in part by the Gordon and Betty Moore Foundation’s Data-Driven Discovery Initiative through Grants [GBMF4552](https://www.moore.org/grant-detail?grantId=GBMF4552) to Casey Greene and [GBMF4560](https://www.moore.org/grant-detail?grantId=GBMF4560) to Blair Sullivan.
