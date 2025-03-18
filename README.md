# OSSFE Tutorial

This repository contains example files and setup instructions for demonstrating Pyccel, a Python-to-Fortran/C accelerator, presented at the **Open Source Software for Fusion Energy (OSSFE)** conference.

---

## What is Pyccel?

[Pyccel](https://github.com/pyccel/pyccel) is a Python package designed to translate performance-critical Python code into optimized, compiled Fortran or C extensions, significantly accelerating numerical computations while retaining the flexibility and ease-of-use of Python.

---

## How to Run the Demo

### 1. Build and Run Docker Container

```bash
docker build -t pyccel_env .
docker run -it -v $(pwd):/workspace pyccel_env
```

### 2. Compile Python Code with Pyccel

Inside your container:

```bash
pyccel ossfe_demo.py
```

### 3. Execute Compiled Code

Run the compiled version seamlessly within Python:

```python
from ossfe_demo import greet_ossfe
greet_ossfe()
```

---

## Resources

- [Pyccel GitHub](https://github.com/pyccel/pyccel)

---

