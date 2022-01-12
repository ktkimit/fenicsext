import os

import dolfin as dl

import fenicsext_path
from fenicsext import NormalDirichletBC


class BottomBoundary(dl.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[1] < dl.DOLFIN_EPS


# Create a mesh.
nsteps = 1
mesh = dl.UnitDiscMesh.create(dl.MPI.comm_world, nsteps, 2, 2)
bottom = BottomBoundary()
boundary_markers = dl.MeshFunction("size_t", mesh, mesh.geometry().dim() - 1)
boundary_markers.set_all(0)
bottom.mark(boundary_markers, 1)
ds = dl.Measure("ds", domain=mesh, subdomain_data=boundary_markers)

# Define the finite element function space.
P1 = dl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
P2 = dl.VectorElement("Lagrange", mesh.ufl_cell(), 2)
TH = P2 * P1
Vh = dl.FunctionSpace(mesh, TH)

# Define the no-penetration boundary condition.
normbc = NormalDirichletBC(Vh, dl.Constant(0.0), boundary_markers, 1)

# Define trial and test functions.
trial = dl.TrialFunction(Vh)
test = dl.TestFunction(Vh)

# Define the weak form.
load = dl.Constant((0.0, -1.0))

normal = dl.FacetNormal(mesh)
u, p = dl.split(trial)
v, q = dl.split(test)

tang_u = u - dl.outer(normal, normal) * u
tang_v = v - dl.outer(normal, normal) * v

a = (
    dl.inner(dl.sym(dl.grad(u)), dl.sym(dl.grad(v))) * dl.dx
    - dl.div(u) * q * dl.dx
    - dl.div(v) * p * dl.dx
    + dl.inner(tang_u, tang_v) * ds(1)
)
b = dl.inner(load, v) * dl.dx

# Assemble the linear system.
# NOTE: The system is rotated such that the nodal dofs on the boundary
# correspond to the (locally) rotated coordinates.
A = dl.assemble(a)
B = dl.assemble(b)
normbc.apply(A, B)

# Solve the linear system.
# NOTE: The solution vector gets back to the original coordinates after solving the
# linear system of equations.
x = dl.Function(Vh)
dl.solve(A, x.vector(), B)
normbc.get_original_vec(x.vector())

figure_path = os.path.dirname(os.path.realpath(__file__)) + "/figures"
dl.File(figure_path + "/u.pvd") << x.sub(0)
dl.File(figure_path + "/p.pvd") << x.sub(0)
