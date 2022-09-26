# coneref

`coneref` is a Python package for iteratively refining an approximate solution to a conic linear program. It has an interface to [`cvxpy`](https://www.cvxpy.org) and is easy to use.

### Installation
`coneref` is available on PyPI. Install it with

```bash
pip install coneref
```
`coneref` uses [`Eigen`](https://eigen.tuxfamily.org/index.php?title=Main_Page) for linear algebra. Eigen has a built-in support for vectorization. To enable vectorization, install `coneref` from the source distribution using the command (on Linux)
```bash
MARCH_NATIVE=1 pip install coneref --no-binary=coneref
```
Enabling vectorization typically decreases the refinement time by a factor 2 for problems with matrix variables and by 10% for problems with vector variables.

Building `coneref` from the source distribution requires a compiler that supports C++11.

### Refinement
The basic idea of the refinement procedure is to reduce the problem of solving a conic linear program to the problem of finding a zero of a nonlinear map. A Newton-based method is then applied to solve the root-finding problem. For more details, see the paper [Solution Refinement at Regular Points of Conic Problems](https://web.stanford.edu/~boyd/papers/cone_prog_refine.html). Also see the note [refinement_theory_note.pdf](https://github.com/dance858/coneref/blob/main/refinement_theory_note.pdf). 


### Basic example
`coneref` can be used in combination with [`cvxpy`](https://www.cvxpy.org). 

The following optimization problem arises in the context of sparse inverse covariance estimation:

$$\begin{equation*} \begin{array}{ll} \text{minimize} & \text{log det}  (S) + \text{Tr} (S Q) + \alpha \||S\||_1 \end{array} \end{equation*},$$

where the optimization variable is $S \in \bf{S}^n$.

### Interface
The package exposes the function
```python
cvxpy_solve(prob, ref_iter=2, lsqr_iter=500, verbose_scs=True, scs_opts={}, warmstart=False, verbose_ref1=True, verbose_ref2=True).
```
Here the arguments are defined as follows.
* `prob` - cvxpy-problem.
* `ref-iter` - number of refinement steps.
* `lsqr_iter` - each refinement step requires solving a sparse linear system approximately. This parameter determines the maximum number of LSQR                     iterations.
* `verbose_scs` - verbose parameter sent to SCS.
* `warm_start` - SCS parameter.
* `verbose_ref1` - If true the refinement algorithm outputs the KKT-residuals.
* `verbose_ref2` - If true the refinement algorithm outputs the norm of the normalized residual map.

The function modifies the object `prob` in the following way: `TODO`


### How well does it work?
The refinement algorithm can often produce a more accurate solution with a small additional cost, see Section 4 of [refinement_theory_note.pdf](https://github.com/dance858/coneref/blob/main/refinement_theory_note.pdf) for empirical results.

### Acknowledgements
`TODO` Some code has been modified from ... 
Also add tests.

### Citing
If you find this package useful, consider citing the following works.

```
@misc{cpr,
    author       = {Cederberg, D. and Boyd, S.},
    title        = {{coneref}: conic LP refinement, version 0.1},
    howpublished = {\url{https://github.com/dance858/coneref}},
    year         = 2022
}

@article{cpr2019,
    author       = {Busseti, E. and Moursi, W. and Boyd, S.},
    title        = {Solution refinement at regular points of conic problems},
    journal      = {Computational Optimization and Applications},
    year         = {2019},
    volume       = {74},
    number       = {3},
    pages        = {627--643},
}
```

