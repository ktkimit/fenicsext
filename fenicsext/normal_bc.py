import numpy as np
import dolfin as dl
from petsc4py import PETSc


class NormalDirichletBC(object):
    """Handling normal type boundary conditions by using coordiates rotation"""

    def __init__(
        self,
        Vh: dl.FunctionSpace,
        bcvalues: dl.Expression,
        boundary_markers: dl.MeshFunction,
        mark: int,
        coord_bc: str = "normal",
    ):

        """Constructor

        Parameters
        ----------
        Vh:
            Function space.
        bcvalues:
            Boundary values for `coord_bc` component.
        boundary_markers:
            Mesh function handling boundary mesh entities.
        mark: int
            Mark of the boundary entities that the boundary conditions will apply on.
        coord_bc: str, optional
            Coordinate component for which the boundary condition is defined.
            "normal" or "tangent".

        """
        self.Vh = Vh

        self._set_vector_space(Vh)

        assert self.Uh.num_sub_spaces() > 1, "Function space should be 2D or 3D."
        if self.Uh.num_sub_spaces() == 3:
            raise NotImplementedError("3D case is not implemented yet.")

        self.bc_dofs = list()
        self.bc_values = list()
        for i in range(self.Uh.num_sub_spaces()):
            bc = dl.DirichletBC(self.Uh.sub(i), bcvalues, boundary_markers, mark)
            self.bc_dofs.append(list(bc.get_boundary_values().keys()))
            self.bc_values.append(list(bc.get_boundary_values().values()))

        self._compute_dofs_nodes()

        ds = dl.Measure("ds", domain=self.Vh.mesh(), subdomain_data=boundary_markers)
        self._compute_normal(ds(mark))

        self._construct_rotation()

        if coord_bc == "normal":
            self.index_bc_apply = 0
        elif coord_bc == "tangent":
            self.index_bc_apply = 1
        else:
            raise NotImplementedError("Wrong input value for " + coord_bc)

    def rotate_matrix(
        self, A: dl.Matrix, apply_bc: bool = False, out_fmt: str = "dolfin"
    ):
        """Rotate matrix

        Concretly this function computes :math:`\mathbf{R}^T \mathbf{A} \mathbf{R}`.

        Parameters
        ----------
        A:
            Input matrix.
        apply_bc: bool
            If True, zero out the columns and rows corresponding to the
            boundary dofs (normal or tangent set by `self.index_bc_apply`) and
            then set the diagonal component to 1.
        out_fmt: str
            Output format. Should be `"dolfin"` or `"petsc"`.
        """
        Apetsc = dl.as_backend_type(A).mat()

        B = Apetsc.matMult(self.Rpetsc)
        C = self.Rpetsc.transposeMatMult(B)

        if apply_bc:
            # Corresponding rows and columns of `C` are zeroed out.
            # NOTE: The diagonal value of the zeroed out row is set to 1, but
            # this leads to ill-conditioned matrix if non-zero components of
            # `C` are very large.
            C.zeroRowsColumns(self.bc_dofs[self.index_bc_apply], diag=1.0)

        if out_fmt == "dolfin":
            A.zero()
            A.axpy(1.0, dl.Matrix(dl.PETScMatrix(C)), False)
        elif out_fmt == "petsc":
            return C
        else:
            raise NotImplementedError("Format " + out_fmt + " is not supported.")

    def rotate_vector(
        self, x: dl.Vector, apply_bc: bool = False, out_fmt: str = "dolfin"
    ):
        """Rotate vector

        Compute :math:`\mathbf{R}^T \mathbf{x}`.

        Parameters
        ----------
        out_fmt: str
            Output format. Should be `"dolfin"` or `"petsc"`
        x:
            Input vector.
        apply_bc: bool
            If True, set the components corresponding to the boundary dofs
            (normal or tangent set by `self.index_bc_apply`) to the boundary
            values.
        out_fmt: str
            Output format. Should be `"dolfin"` or `"petsc"`.

        """
        xpetsc = dl.as_backend_type(x).vec()

        y = self.Rpetsc.createVecRight()
        self.Rpetsc.multTranspose(xpetsc, y)

        if apply_bc:
            # Corresponding components of `y` are set to the boundary values.
            y.setValues(
                self.bc_dofs[self.index_bc_apply], self.bc_values[self.index_bc_apply]
            )

        if out_fmt == "dolfin":
            x.zero()
            x.axpy(1.0, dl.Vector(dl.PETScVector(y)))
        elif out_fmt == "petsc":
            return y
        else:
            raise NotImplementedError("Format " + out_fmt + " is not supported.")

    def apply(self, *args):
        """Apply the boundary conditions"""
        if len(args) == 1 and isinstance(args[0], dl.Vector):
            self.rotate_vector(args[0], apply_bc=True, out_fmt="dolfin")
        elif len(args) == 1 and isinstance(args[0], dl.Matrix):
            self.rotate_matrix(args[0], apply_bc=True)
        elif len(args) == 2:
            if isinstance(args[0], dl.Matrix) and isinstance(args[1], dl.Vector):
                self.rotate_matrix(args[0], apply_bc=True, out_fmt="dolfin")
                self.rotate_vector(args[1], apply_bc=True, out_fmt="dolfin")
            else:
                raise NotImplementedError(
                    "No overloaded functions for given arguments."
                )
        else:
            raise NotImplementedError("No overloaded functions for given arguments.")

    def get_original_vec(self, x: dl.Vector):
        """Get back to the original dofs

        Compute :math:`\mathbf{R} \mathbf{x}`.

        Parameters
        ----------
        x: dl.Vector
            Input vector.

        """
        xpetsc = dl.as_backend_type(x).vec()
        y = self.Rpetsc.createVecLeft()
        self.Rpetsc.mult(xpetsc, y)

        x.zero()
        x.axpy(1.0, dl.Vector(dl.PETScVector(y)))

    def _set_vector_space(self, Vh):
        """Set the function space for the `VectorElement` from `Vh`

        Parameters
        ----------
        Vh: Input function space.
        """
        element = Vh.element().signature().split('(')[0]
        if element == 'VectorElement':
            self.Uh = Vh
            self.uh_index = None
            return
        elif element == 'MixedElement':
            for i in range(Vh.num_sub_spaces()):
                if Vh.sub(i).element().signature().split('(')[0] == 'VectorElement':
                    self.Uh = Vh.sub(i)
                    self.uh_index = i
                    return

        raise NotImplementedError("Wrong function space input.")

    def _compute_dofs_nodes(self):
        """Compute coordinates and dofs of each node on the boundary.

        self.coords_nodes[i,:] = coordinates of node i
        self.dofs_nodes[i,:] = dofs in x, y, z directions of node i

        """
        # make 'self.bc_dofs' as set type
        bc_dofs = list()
        for i in range(self.Uh.num_sub_spaces()):
            bc_dofs.append(set(self.bc_dofs[i]))

        # coordinates of the nodes living on the boundary
        self.coords_nodes = self.Vh.tabulate_dof_coordinates()[self.bc_dofs[0]]

        # search for dof pairs for each node
        # PERF: Is there better way to do this or some built-in FEniCS
        # functions for this?
        tol = 1.0e-16
        ndofs = len(self.bc_dofs[0])
        self.dofs_nodes = np.empty((ndofs, self.Uh.num_sub_spaces()), dtype=np.int32)

        tree = self.Vh.mesh().bounding_box_tree()
        for ix in range(ndofs):
            dofx = self.bc_dofs[0][ix]
            self.dofs_nodes[ix, 0] = dofx
            px = dl.Point(self.coords_nodes[ix, :])

            for n in range(1, self.Uh.num_sub_spaces()):
                cells = tree.compute_collisions(px)
                dofmap = self.Uh.sub(n).dofmap()

                for cell in cells:
                    dofs_cell = dofmap.cell_dofs(cell)
                    for dof in dofs_cell:
                        if dof in bc_dofs[n]:
                            pn = dl.Point(self.Vh.tabulate_dof_coordinates()[dof])
                            distance = px.distance(pn)
                            if distance < tol:
                                self.dofs_nodes[ix, n] = dof
                                break
                    else:
                        continue
                    break

    def _compute_normal(self, ds):
        """Compute normal vector of the boundary nodes.

        `self.normal` is outward, but not the unit vector.

        Parameters
        ----------
        ds: the measure of the boundary for which the normal is computed.

        """
        n = dl.FacetNormal(self.Vh.mesh())
        u = dl.TrialFunction(self.Vh)
        v = dl.TestFunction(self.Vh)

        if self.uh_index is not None:
            u = dl.split(u)[self.uh_index]
            v = dl.split(v)[self.uh_index]

        a = dl.inner(u, v) * ds
        b = dl.inner(n, v) * ds
        A = dl.assemble(a, keep_diagonal=True)
        B = dl.assemble(b)

        A.ident_zeros()
        self.normal = dl.Function(self.Vh)
        dl.solve(A, self.normal.vector(), B)

    def _construct_rotation(self):
        # WARNING: Only 2D case is implemented.
        """Construct rotation matrix `self.Rpetsc`."""
        n = self.Vh.dim()
        self.Rpetsc = PETSc.Mat().createAIJ([n, n])
        self.Rpetsc.setUp()

        x = PETSc.Vec().createSeq(n)
        x.set(1.0)
        self.Rpetsc.setDiagonal(x)

        for i in range(self.dofs_nodes.shape[0]):
            dofx = self.dofs_nodes[i, 0]
            dofy = self.dofs_nodes[i, 1]

            nx = self.normal.vector()[dofx]
            ny = self.normal.vector()[dofy]

            norm2 = np.sqrt(nx ** 2 + ny ** 2)
            nx /= norm2
            ny /= norm2

            # ux = nx*un + tx*ut
            # uy = ny*un + ty*ut
            #
            # (ux, uy): vector components in x, y directions
            # (un, ut): vector components in normal, tangent directions
            # (nx, ny): unit normal vector
            # (tx, ty): unit tangent vector
            self.Rpetsc[dofx, dofx] = nx
            self.Rpetsc[dofx, dofy] = -ny
            self.Rpetsc[dofy, dofx] = ny
            self.Rpetsc[dofy, dofy] = nx

        self.Rpetsc.assemble()

