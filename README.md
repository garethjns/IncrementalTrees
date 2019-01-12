## Installing package

For details refer to Python Packaging User Guide's [Packaging Projects](https://packaging.python.org/tutorials/packaging-projects/) and [Installing Packages](https://packaging.python.org/tutorials/installing-packages/)

Quick start:

- python -m pip install --upgrade pip setuptools wheel
- cd into directory where setup.py is located.
- python3 setup.py sdist bdist_wheel
- a folder named "dist" will be created, and ".tar.gz" + ".whl" files will be created there, where the ".tar.gz" should be used to do a pip install for python (or a virtual environment)
