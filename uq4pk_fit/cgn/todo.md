# TODOs

- Implement the following interface:
```python
from uq4pk_fit.cgn import *

x = Parameter(dim=20, name='x')
y = Parameter(dim=10, name='y')
x.alpha = 10   # changes the regularization parameter (default is no regularization)
x.regop = P     # sets the regularization operator
x.mean = np.ones(20)    # sets the mean equal to the one-vector
x.lb = np.zeros(20)     # set lower bound for x
eqcon = LinearConstraint(a=a, b=b, parameters=[x, y], ctype="eq")   # defines an equality constraint both on x and y
# Generate new problem:
newproblem = Problem(fun=myfun, jac=myjac, parameters=[x, y], constraints=[eqcon])
# You can also add constraints afterwards:
newproblem.constraints.append(incon)
# And even adapt the regularization
newproblem.parameter("x").beta = 1
# initialize solver
solver = CGN()
# set solver options
solver.options.maxiter = 42
# solve the problem
solution = solver.solve(problem=newproblem, start=[x0, y0])
# now, the minimizers can be accessed via
x_opt = solution.minimizer("x")
y_opt = solution.minimizer("y")
```