<h1 align="center">
  NAVLIB
  
  <div align="center">
  
  ![GitHub Repo stars](https://img.shields.io/github/stars/sturdivant20/navlib.py)
  [![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
  ![GitHub pull requests](https://img.shields.io/github/issues-pr/sturdivant20/navlib.py)
  ![GitHub issues](https://img.shields.io/github/issues/sturdivant20/spydr)
  ![GitHub contributors](https://img.shields.io/github/contributors/sturdivant20/navlib.py)
    
  </div>
  
</h1>

<h3 align="center">Useful tools for navigation purposes along with a few predefined filter applications</h3>
<h4 align="center">Daniel Sturdivant &ltsturdivant20@gmail.com&gt</h4>

# Docs

- [TODO](#todo)
- [Installation](#installation)
- [Examples](#running-examples)

## TODO
1. Figure out why numba fails to complile least-squares code.
2. Explore deeply integrated models.

## Installation
Clone the project into desired folder and `cd` into it:
```shell
git clone git@github.com:sturdivant20/navlib.py.git navlib
cd navlib
```

Create a virtual environment or use your base Python environment. For creating a virtual 
environment, refer to 
[this article](https://www.freecodecamp.org/news/how-to-setup-virtual-environments-in-python/). 
Ensure the virtual environment is active in your editor if you decide to use one (***I highly recommend this***).

From here, `pip install` the spydr package in editable mode.
```shell
pip install -e .
```
Or system install with.
```shell
pip install .
```

## Running Examples
To run make sure you are out of the 'navlib' directory (you may need to run `cd ..`) and run the receiver with the following command.
```shell
python3 navlib/examples/test_coordinates.py
python3 navlib/examples/test_filters.py
```
You will need a valid config file for this to work.