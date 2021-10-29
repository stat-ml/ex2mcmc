# Ex$^2$MCMC: Sampling through Exploration Exploitation

## Table of content

- [Ex$^2$MCMC: Sampling through Exploration Exploitation](#ex2mcmc-sampling-through-exploration-exploitation)
  - [Table of content](#table-of-content)
  - [Installation](#installation)
    - [With docker](#with-docker)
    - [Or without docker](#or-without-docker)
  - [Some results](#some-results)
  - [Acknoledgements](#acknoledgements)

## Installation

### With docker
Build docker image

```bash
docker build --tag ex2mcmc:v1 .
```

Run docker container

```bash
docker run -dit --name ex2mcmc ex2mcmc:v1
```

### Or without docker

```bash
conda create -y --name ex2mcmc python=3.8

conda activate ex2mcmc
```

```bash
pip install poetry
```

```bash
poetry install
```

## Some results


## Acknoledgements
