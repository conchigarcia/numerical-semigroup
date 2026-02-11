# NumericalSemigroup

A Python library designed for the analysis of **numerical semigroups**, focusing on computation of algebraic invariants, factorizations and interactive visualization.

## Repository contents

* [`NumericalSemigroup.py`](NumericalSemigroup.py): the main file containing the class definition and all the methods to compute invariants, factorizations, and visualizations.
* [`Tutorial.ipynb`](Tutorial.ipynb): a Jupyter Notebook that explains how to define a numerical semigroup and demonstrates how to use all the methods implemented in the class.
* [`requirements.txt`](requirements.txt): a text file listing all the necessary libraries to execute the code.

## Installation

To use the code, clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage Example
```python
from NumericalSemigroup import *

# Define a semigroup S = <4, 6, 9>
S = NumericalSemigroup(4, 6, 9)

# Calculate invariants
print(S.frobenius_number())
print(S.genus())

# Check Tutorial.ipynb for more examples.
````

## References

The code structure and algorithms were inspired by the following packages:

* [NumericalSgps (GAP package)](https://github.com/gap-packages/numericalsgps?tab=readme-ov-file#papers-using-numericalsgps): serves as the primary reference for the algorithmic logic, currently maintained by M. Delgado and P.A. García-Sánchez.

* [numericalSemigroupPackage (Python package)](https://github.com/gilad-moskowitz/numericalSemigroupPackage): provides insights into code structure and Python adaptation, developed by Gilad Moskowitz.