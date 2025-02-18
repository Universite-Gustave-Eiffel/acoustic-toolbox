# Guide for Contributors

## How to contribute to this project

Awaiting content...

## Documentation

### Docstrings

Docstrings are used throughout to document the code. They are written in the [Google style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html). 

When including mathematical equations in docstrings, use LaTeX formatting and **prefix the docstring with `r`** to indicate a raw string. For example:

```python
def quadratic(a: float, b: float, c: float):
    r"""
    Calculate the roots of a quadratic equation.

    The roots are given by the formula:
        $$
        x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}
        $$

    Args:
        a: The coefficient of the quadratic term.
        b: The coefficient of the linear term.
        c: The constant term.

    Returns:
        tuple: The roots of the quadratic equation.
    """
    ...
```
