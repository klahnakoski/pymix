pymix
=====

Refactoring of PyMix so it is pure Python:

* Split code into smaller files
* Remove GSL GNU Scientific library dependency
* Remove GHMM for mixtures of HMMs
* Remove Numpy dependency (well, not yet)

Requirements
------------

**Required Packages**

* Python (version 2.7.x recommended)
* SciPy (and NumPy, of course)

**Optional Packages**

* pylab for plotting functions in plotMixture.py

Installation
------------

    git clone https://github.com/klahnakoski/pymix.git

Run Tests
---------

From the `pymix` directory:

	set PYTHONPATH=.
	python -m unittest discover -s C:\Users\kyle\code\pymix\tests

Documentation
--------------

Example code for most aspects of the library can be found in
the `/examples` subdirectory and `tests/mixtureunittest.py`.
Documentation from the original project is available on the
Pymix home page www.pymix.org.

