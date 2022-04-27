# Fine-tuning reproduciblity of LIGO Black Hole signal tutorial, Part II

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/UCB-stat-159-s22/hw06-minorijaggia.git/HEAD?labpath=index.ipynb)

https://mybinder.org/v2/gh/UCB-stat-159-s22/hw06-minorijaggia.git/HEAD?labpath=index.ipynb


**Note:** This repository is public so that Binder can find it. All code and data is based on the original [LIGO Center for Open Science Tutorial Repository](https://github.com/losc-tutorial/LOSC_Event_tutorial). This repository is a class exercise that restructures the original LIGO code for improved reproducibility, as a homework assignment for the [Spring 2022 installment of UC Berkeley's Stat 159/259 course, _Reproducible and Collaborative Data Science_](https://ucb-stat-159-s22.github.io). Authorship of the original analysis code rests with the LIGO collaboration.


**Makefile Targets:** 

`env`: creates and activates environment for index.ipynb

`html`: build the JupyterBook if repo is cloned locally, or is accessed with the VNC desktop on the hub

`html-hub`: builds the JupyterBook so that you can view it on the hub

`clean`: remove the figures, audio files and _build folders


**Testing the Ligotools Package:**

Run `pytest ligotools` in the repo's main directory using the ligo environment

**Building the Jupyter Book:**

In order to build the Jupyter Book, you must be in the regular notebook environment. Deactivate the ligo environment using `conda deactivate` prior to running `make hub` or `make html-hub`

Click this link to access the public Jupyter Book: 
https://ucb-stat-159-s22.github.io/hw06-minorijaggia/