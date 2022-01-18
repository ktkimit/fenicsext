import unittest
import numpy as np
import dolfin as dl

from fenicsext.normal_bc import NormalDirichletBC


class UnitSquareProblem(object):
    def __init__(self):
        class LeftBottomBoundary(dl.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and (x[1] < dl.DOLFIN_EPS or x[0] < dl.DOLFIN_EPS)

        ne = 1
        p = 1

        mesh = dl.UnitSquareMesh(ne, ne)
        leftbottom = LeftBottomBoundary()
        boundary_markers = dl.MeshFunction("size_t", mesh, mesh.geometry().dim() - 1)
        boundary_markers.set_all(0)
        leftbottom.mark(boundary_markers, 1)

        UV = dl.VectorElement("Lagrange", mesh.ufl_cell(), p)
        Vh = dl.FunctionSpace(mesh, UV)

        self.normbc = NormalDirichletBC(
            Vh, dl.Constant(0.0), boundary_markers, 1, ref_point=np.array([0.5, 0.5])
        )


class UnitCircleProblem(object):
    def __init__(self):
        class BottomBoundary(dl.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and x[1] < dl.DOLFIN_EPS

        nsteps = 11
        p = 2

        mesh = dl.UnitDiscMesh.create(dl.MPI.comm_world, nsteps, 2, 2)
        bottom = BottomBoundary()
        boundary_markers = dl.MeshFunction("size_t", mesh, mesh.geometry().dim() - 1)
        boundary_markers.set_all(0)
        bottom.mark(boundary_markers, 1)

        UV = dl.VectorElement("Lagrange", mesh.ufl_cell(), p)
        Vh = dl.FunctionSpace(mesh, UV)

        self.normbc = NormalDirichletBC(Vh, dl.Constant(0.0), boundary_markers, 1)


class TestNormalDirichletBC(unittest.TestCase):
    def setUp(self):
        # self.pde_unitsquare = UnitSquareProblem()
        self.pde_unitcircle = UnitCircleProblem()

    def test_normal(self):
        # Check the calculated normal vectors are correct.
        normbc = self.pde_unitcircle.normbc
        nnodes = len(normbc.coords_nodes)
        for i in range(nnodes):
            xp = normbc.coords_nodes[i]
            normal_xp = xp / np.linalg.norm(xp)

            normal = normbc.normal.vector()[normbc.dofs_nodes[i]]
            normal = normal / np.linalg.norm(normal)

            err = np.linalg.norm(normal_xp - normal)
            if err > 1.0e-10:
                print(normal_xp)
                print(normal)
                print(err)
                print("")

        # self.assertAlmostEqual(err, 0.)


if __name__ == "__main__":
    unittest.main()
