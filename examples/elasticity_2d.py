"""
This example considers a 2D elastic body in plan stress condition subjected to
a unit body load. The purpose of this example is to show how to impose a
no-penetration boundary condition using local (nodal) coordinate transform.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath("")))

import numpy as np
import dolfin as dl
from petsc4py import PETSc

from fenicsext import NormalDirichletBC


class BottomBoundary(dl.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[1] < dl.DOLFIN_EPS


# Set material properties.
E = dl.Constant(1e6)
nu = dl.Constant(0.3)
f = dl.Constant((0.0, 1.0e3))

# Create a mesh.
nsteps = 10
mesh = dl.UnitDiscMesh.create(dl.MPI.comm_world, nsteps, 2, 2)
bottom = BottomBoundary()
boundary_markers = dl.MeshFunction("size_t", mesh, mesh.geometry().dim() - 1)
boundary_markers.set_all(0)
bottom.mark(boundary_markers, 1)
ds = dl.Measure("ds", domain=mesh, subdomain_data=boundary_markers)

# Define the finite element function space.
UV = dl.VectorElement("Lagrange", mesh.ufl_cell(), 1)
Vh = dl.FunctionSpace(mesh, UV)

# Define the no-penetration boundary condition.
normbc = NormalDirichletBC(Vh, dl.Constant(0.0), boundary_markers, 1)

# Define trial and test functions.
u = dl.TrialFunction(Vh)
v = dl.TestFunction(Vh)

# Define the weak form.
eps_u = dl.sym(dl.grad(u))
eps_v = dl.sym(dl.grad(v))
sigma_v = E / (1 + nu) * (eps_v + nu / (1 - nu) * dl.tr(eps_v) * dl.Identity(2))
a = dl.inner(sigma_v, eps_u) * dl.dx
l = dl.inner(f, v) * dl.dx

# Assemble the linear system.
# NOTE: The system is rotated such that the nodal dofs on the boundary
# correspond to the (locally) rotated coordinates.
A = dl.assemble(a)
L = dl.assemble(l)
normbc.apply(A, L)

# Solve the linear system.
# NOTE: The solution vector gets back to the original coordinates after solving the
# linear system of equations.
x = dl.Function(Vh)
dl.solve(A, x.vector(), L)
normbc.get_original_vec(x.vector())

dl.File("./results/utrue.pvd") << x
