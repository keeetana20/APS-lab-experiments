# Heart Disease ML Pipeline

A complete, end to end machine learning pipeline that runs entirely on your
own machine, with continuous integration and continuous deployment through
GitHub Actions.

The pipeline downloads the UCI Heart Disease dataset, cleans it, engineers
features, trains a PyTorch neural network, evaluates it, deploys it to a
local model server, and monitors that server's predictions for drift. Every
stage is a plain Python script that reads its input from a local folder and
writes its output to another local folder. You can run the whole thing,
inspect any intermediate file, or rerun a single stage, all without any
external service.

## Prerequisites

You need the following installed before you start. None of them require a
paid account or a GPU.

- **Git**, to clone this repository.
- **Python 3.10, 3.11, or 3.12**. Run `python3 --version` (or `python
  --version` on Windows) to check what you have. If you use conda instead,
  it can create an environment with the right Python version for you, so a
  separate system-wide Python install is not required in that case.
- **pip**, which ships with Python and with every conda environment.
- **make**, to run the shortcuts in the Makefile. Linux and macOS have it
  by default. On Windows, install it through WSL or Git Bash, or skip it
  and run the underlying `python -m src.<module>` commands directly,
  listed under "Running each stage individually" below.
- An **internet connection**, needed once, to download the dataset.

A GPU is optional. Every stage runs fine on CPU.

## Setup

First clone the repository.

```bash
git clone <this-repository-url>
cd ml-pipeline-local
```

Then create an isolated Python environment using either `venv` or `conda`.
Both give the same result. Use whichever you already have.

### Option A: venv

```bash
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

On Windows, activate the environment with `venv\Scripts\activate` instead
of `source venv/bin/activate`.

On Debian or Ubuntu, creating a virtual environment can fail with
"ensurepip is not available." If that happens, install the matching
`python3.X-venv` package first, for example `sudo apt install
python3.12-venv`, then retry the `python3 -m venv venv` command.

### Option B: conda

```bash
conda create -n ml-pipeline-local python=3.12
conda activate ml-pipeline-local

pip install -r requirements.txt
```

This project's dependencies are only published on PyPI, not on any conda
channel, so `pip install -r requirements.txt` is still the right command
here too. Conda is only being used to create and manage the Python
environment itself, the same role `venv` plays in Option A.

Any of 3.10, 3.11, or 3.12 works in place of 3.12 in that command.

### After either option

You must activate this environment in every new terminal window before
running any command in this README. With venv that means running the
`source venv/bin/activate` line again. With conda it means running `conda
activate ml-pipeline-local` again. If a command fails with "module not
found," activating the environment in that terminal is usually the fix.

## Getting started

Run the full pipeline once, end to end:

```bash
make pipeline
```

This downloads the dataset, preprocesses it, trains a model, evaluates it,
and deploys it. It takes a few minutes on CPU.

Then start the local model server and ask it for a prediction:

```bash
make serve      # keep this running in its own terminal
make predict    # in a second terminal, with the virtual environment activated
```

