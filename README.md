QuTiP notebooks
===============

These files are [IPython notebooks](http://ipython.org/notebook.html) for
testing different parts of QuTiP. These tests serves a somewhat different
purpose than the unit test suite that is installed as a part of QuTiP. Instead
of being small isolated (unit) tests, these notebooks are often more like
integration tests, which exercise a larger part of the QuTiP codebase to make
sure that different parts work together as expected, or tests that exercise
various related parts in a module in a single location.

To open these files, start an IPython notebook server by running the following
command in the directory that contains the files:

    $ jupyter notebook

or, if you have an old version of IPython installed

    $ ipython notebook

This will open a new page in your web browser, showing the IPython notebook
dashboard page with an index of all the notebooks.

Older notebooks are in v3 format. The newer notebooks are in v4 format.
If you are using a version of IPython notebook that does not support v4 format.
(which would be v3.0.0 or lower), then you can convert these notebooks using:

    $ jupyter nbconvert --to notebook --nbformat 3 <nb_to_convert>

# Interactive online versions

This is currently (Jul 2019) running on the host service provided by My Binder.
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/qutip/qutip-notebooks/master?filepath=index.ipynb)

# Online read-only versions

You can also view the notebooks online, as read-only HTML pages rendered by
http://nbviewer.ipython.org. The notebooks are rendered also by My Binder. [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/qutip/qutip-notebooks/master?filepath=index.ipynb).

# Tutorials and lectures

The notebooks are organized with tutorials and lectures on [index.ipynb](index.ipynb).

# Contribute

You are most welcome to contribute to the QuTiP notebooks  by forking this
repository and sending pull requests, or filing bug reports at the
[issues page](http://github.com/qutip/qutip-notebooks/issues), or send us bug reports,
questions, or your proposed changes to our
[QuTiP discussion group](http://groups.google.com/group/qutip).

All contributions are acknowledged in the
[contributors](http://github.com/qutip/qutip-doc/blob/master/contributors.rst)
section in the documentation.

Note that all notebook contributions must have a ```qutip.about()``` line at the end for reproducibility purposes. It is also encouraged to add the notebook to the [tutorial page](http://qutip.org/tutorials.html) and the [index](index.ipynb).

For more information, including technical advice, please see [Contributing to QuTiP development](https://github.com/qutip/qutip-doc/blob/master/CONTRIBUTING.md).
