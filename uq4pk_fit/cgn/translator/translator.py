
import numpy as np
from typing import List, Literal

from uq4pk_fit.cgn.cnls_solve import CNLS, CNLSConstraint, ConcreteConstraint, NullConstraint
from uq4pk_fit.cgn.cnls_solve.cnls_solution import CNLSSolution, OptimizationStatus

from uq4pk_fit.cgn.problem.linear_constraint import LinearConstraint
from uq4pk_fit.cgn.translator.get_sub_matrix import get_sub_matrix
from uq4pk_fit.cgn.problem.problem import Problem
from .multiparameter import MultiParameter
from .translated_solution import TranslatedSolution


class Translator:
    """
    Translates a cgn.Problem object to a CNLS object.
    """
    def __init__(self, problem: Problem):
        self._problem = problem
        self._nparams = problem.nparams
        # read the problem parameters into a Parameters object
        self._multi_parameter = MultiParameter(self._problem._parameter_list)

    def translate(self) -> CNLS:
        """
        Returns a CNLS equivalent to the :py:class``Problem`` object.
        """
        fun = self._modify_function(self._problem.fun)
        jac = self._modify_function(self._problem.jac)
        q = self._problem.q
        eqcon = self._combine_constraints(ctype="eq")
        incon = self._combine_constraints(ctype="ineq")
        mean = self._multi_parameter.mean
        r = self._multi_parameter.regop
        lb = self._multi_parameter.lb
        ub = self._multi_parameter.ub
        scale = self._problem.scale
        cnls = CNLS(func=fun, jac=jac, q=q, r=r, m=mean, eqcon=eqcon, incon=incon, lb=lb, ub=ub, scale=scale)
        return cnls

    def _modify_function(self, func: callable):
        """
        Takes function that takes list of arguments and transforms it to function that takes concatenated
        vector as input.
        :param func: function that takes a tuple as argument
        :return: function that takes a single vector as argument
        """
        def newfunc(x):
            x_tuple = self._extract_x(x)
            return func(*x_tuple)
        return newfunc

    def _extract_x(self, x):
        """
        From a concatenated vector, extracts the tuple of parameters
        """
        return self._multi_parameter.extract_x(x)

    def combine_x(self, x_list):
        assert len(x_list) == self._nparams
        return np.concatenate(x_list)

    def _combine_constraints(self, ctype: Literal["eq", "ineq"]) -> CNLSConstraint:
        """
        Reads all equality constraints from self.problem and returns one constraint for the concatenated vector.
        Might be the null constraint.
        """
        # Get all constraints of given ctype from self._problem as list.
        constraint_list = self._get_constraints(ctype=ctype)
        # If the list is empty, return a NullConstraint.
        if len(constraint_list) == 0:
            combined_constraint = NullConstraint(dim=self._problem.n)
        # Else, return the concatenated constraint.
        else:
            # First, we have to formulate all constraints with respect to the concatenated parameter vector.
            list_of_enlarged_constraints = []
            for constraint in constraint_list:
                enlarged_constraint = self._enlarge_constraint(constraint=constraint, ctype=ctype)
                list_of_enlarged_constraints.append(enlarged_constraint)
            # Then, we concatenate constraints
            combined_constraint = self._concatenate_constraints(list_of_enlarged_constraints)
        return combined_constraint

    @staticmethod
    def _concatenate_constraints(list_of_constraints: List[ConcreteConstraint]) -> ConcreteConstraint:
        """
        Given a list of :py:class:`ConcreteConstraint? objects, returns a ConcreteConstraint that represents the
        concatenated constraint.
        """
        a_list = []
        b_list = []
        dim = list_of_constraints[0].dim
        for constraint in list_of_constraints:
            a_list.append(constraint.a)
            b_list.append(constraint.b)
        a_conc = np.concatenate(a_list, axis=0)
        b_conc = np.concatenate(b_list, axis=0)
        concatenated_constraint = ConcreteConstraint(dim=dim, a=a_conc, b=b_conc)
        return concatenated_constraint

    def _get_constraints(self, ctype: Literal["eq", "ineq"]) -> List[LinearConstraint]:
        """
        Returns all constraints of self._problem with the given ctype.
        """
        constraint_list = []
        for constraint in self._problem.constraints:
            if constraint.ctype == ctype:
                constraint_list.append(constraint)
        return constraint_list

    def _enlarge_constraint(self, constraint: LinearConstraint, ctype: Literal["eq", "ineq"]) -> ConcreteConstraint:
        """
        Given an object of type :py:class:`LinearConstraint`, returns the equivalent object of type
        :py:class:`ConcreteConstraint` formulated with respect to the concatenated parameter vector.
        """
        # Get the parameters that the constraint depends on
        parameters = constraint.parameters
        # Initialize the enlarged matrix
        a_enlarged = np.zeros((constraint.cdim, self._problem.n))
        # Write the splitted matrices in the enlarged matrix at the right positions
        for i in range(len(parameters)):
            a_i = get_sub_matrix(constraint, i)
            name = parameters[i].name
            j_i = self._multi_parameter.position_by_name(name)
            k_i = a_i.shape[1]
            a_enlarged[:, j_i:j_i + k_i] = a_i
        # Create a ConcreteConstraint object from a_enlarged
        concrete_constraint = ConcreteConstraint(dim=self._problem.n, a=a_enlarged, b=constraint.b)
        return concrete_constraint

    def translate_solution(self, cnls_solution: CNLSSolution) -> TranslatedSolution:
        """
        Translates the solution of the CNLS problem to the solution of the original, multi-parameter problem
        """
        xmin = self._multi_parameter.extract_x(cnls_solution.minimizer)
        precision = cnls_solution.precision
        cost = cnls_solution.min_cost
        niter = cnls_solution.niter
        success = (cnls_solution.status == OptimizationStatus.converged)
        problem_solution = TranslatedSolution(parameters=self._problem._parameter_list, minimizers=xmin,
                                              precision=precision, cost=cost, success=success, niter=niter)
        return problem_solution
