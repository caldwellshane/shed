# shed
A place to keep tools. Here we're collecting functions for understanding how quantum computers work.

# Installing
Conda makes it really easy to install the thorniest dependency, `qutip`, so there's a `conda-env.yml`
file to support that, eg. via
```
$ conda env create -n shed -f <shed-directory>/conda-env.yml
```
We then make an editable install of `shed`, ie.
```
$ pip install --no-deps -e <shed-directory>
```
