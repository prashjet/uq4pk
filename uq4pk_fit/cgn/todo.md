# TODOs

- Implement the following interface:
```python
x = Parameter(dim=20)
y = Parameter(dim=10)
x.alpha = 10   # changes the regularization parameter (default is no regularization)
x.regop = P     # sets the regularization operator
x.mean = np.ones(20)    # sets the mean equal to the one-vector
x.lb = np.zeros(20)     # set lower bound for x
eqcon = LinearEqualityConstraint(mat=a, rhs=b, parameters=[x, y])   # defines an equality constraint both on x and y
# Generate new problem:
newproblem = Problem(fun=myfun, jac=myjac, parameters=[x, y], constraints=[eqcon])
# initialize solver
solver = CGN()
# set solver options
solver.options.maxiter = 42
# solve the problem
solution = solver.solve(problem=newproblem, start=[x0, y0])
# now, the minimizers can be accessed via
x_opt = x.minimizer
y_opt = y.minimizer
```