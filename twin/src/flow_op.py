# *************************************************************************************************
# Construct linear operators for advection, diffusion, and time step.
# Michael S. Emanuel
# 2024-10-16
# *************************************************************************************************

import time
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.sparse import coo_array, csr_array, diags, identity
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import cg

# Local imports
from utils import Geometry

# Define the types of the data
NumpyFloatArray = npt.NDArray[np.float64]
NumpyBoolArray = npt.NDArray[np.bool_]
NumpyInt8Array = npt.NDArray[np.int8]
NumpyInt32Array = npt.NDArray[np.int32]

# *************************************************************************************************
def calc_dt(velocity: NumpyFloatArray, h: float, diff: float, cfl: float) -> float:
    """Calculate the time step from the velocity field and desired CFL parameter"""
    # Alias diffusion coefficient
    D: float = diff
    # Get the sum of velocity magnitudes
    v: float = np.nanmax(np.sum(np.abs(velocity), axis=-1))
    # Simple calculation, ignoring diffusion
    # dt_lin: float = cfl * h / v
    # Correct for diffusion; f(t) = v * t + [D * t]^1/2. We want f(dt) = cfl * h.
    # Convert this to a quadratic with the change of variables x = dt^2. Obtain
    # v * x^2 + D * x - cfl * h = 0
    x = (np.sqrt(D + 4.0 * cfl * v * h) - np.sqrt(D)) / (2.0 * v)
    dt = x * x
    # Report results
    # print('Time step calculation:')
    # print(f'v = {v:0.3e}')
    # print(f'dt_lin = {dt_lin:0.3e}')
    # print(f'dt_quad = {dt:0.3e}')
    return dt

# *************************************************************************************************
def make_advection_op(velocity: NumpyFloatArray, geom: Geometry, dt: float) -> csr_array:
    """
    Build an NxN advection matrix as a sparse matrix (COO format)
    INPUTS:
        velocity - velocity field; shape (nx, ny, nz, 3)
        geom - geometry object with grid information
        dt - time step
    """
    # Unpack the dimensions from the geometry
    nx: int = geom.nx
    ny: int = geom.ny
    nz: int = geom.nz
    # Shape of the grid
    grid_shape: tuple[int, int, int] = (nx, ny, nz)
    # Unpack the grid row numbers
    rr: NumpyInt32Array = geom.rr
    # Unpack the grid spacing in cm
    h: float = geom.h
    # Construct mask for live cells
    is_live: NumpyBoolArray = geom.is_fluid & ~geom.is_inlet

    # Number of rows in the SOC vector
    N: int = nx * ny * nz

    # Use (i, j, k) indexing for velocity and cell_type
    # Use (r, s) indexing for the advection operator A
    # The advection operator is given by
    # A = - (u dot grad) = - (u_x d/dx + u_y d/dy + u_z d/dz)

    # Upper bound on number of non-zero entries
    nnz_max: int = 6 * N
    # Counter for non-zero entries
    nnz: int = 0

    # An arbitrary entry in the advection matrix is at position (r, s) with value v
    # A[ row[p], col[p] ] = val[p]
    # Numpy array with row indices (r) of the non-zero entries
    row: NumpyInt32Array = np.zeros(nnz_max, dtype=np.int32)
    # Numpy array with column indices (s) of the non-zero entries
    col: NumpyInt32Array = np.zeros(nnz_max, dtype=np.int32)
    # Numpy array with values of the non-zero entries
    val: NumpyFloatArray = np.zeros(nnz_max, dtype=np.float64)
    # Current slice of the arrays
    slc: slice

    # Iterate over coordinates
    for c in range(3):
        # The number of slices in this coordinate, e.g. nx when c=0
        nc: int = grid_shape[c]
        # The velocity c-th components on this slice; shape e.g. (ny, nz) when c=0
        vc: NumpyFloatArray
        # Mask of live cells on this slice
        mask_live: NumpyBoolArray
        # The row index of the current slice; shape e.g. (ny, nz) when c=0
        r: NumpyInt32Array
        # The row index of the slice with x down; shape (ny, nz) when c=0
        s_lo: NumpyInt32Array
        # The row index of the slice with x up; shape (ny, nz) when c=0
        s_hi: NumpyInt32Array
        # The row index of the cell interacting with the current slice; shape e.g. (ny, nz) when c=0
        s: NumpyInt32Array
        # The diagonal entry of the advection operator on a slice; shape e.g. (ny, nz) when c=0
        diag_entry: NumpyFloatArray        

        # Iterate over slices of this coordinate
        for i in range(nc):
            # Get the velocity component and liquid mask for this slice
            if c == 0:
                vc = velocity[i, :, :, c]
                mask_live = is_live[i, :, :]
            elif c == 1:
                vc = velocity[:, i, :, c]
                mask_live = is_live[:, i, :]
            elif c == 2:
                vc = velocity[:, :, i, c]
                mask_live = is_live[:, :, i]

            # Value of i used in the stencil alongside either i_lo or i_hi; usually same as i except on the edges
            i_mi: int = int(np.clip(i, 1, nc-2))
            # Value of i to use when shifting x down and up
            i_lo: int = i_mi - 1
            i_hi: int = i_mi + 1
            # Calculate r, s_lo, s_hi based on the coordinate
            # r is the cell that is changing under advection. 
            # s_lo / s_hi are cells used in the stencil to compute ds/dx.
            if c == 0:
                r    = rr[i   , :, :]
                s_lo = rr[i_lo, :, :]
                s_hi = rr[i_hi, :, :]
            elif c == 1:
                r    = rr[:, i   , :]
                s_lo = rr[:, i_lo, :]
                s_hi = rr[:, i_hi, :]
            elif c == 2:
                r    = rr[:, :, i   ]
                s_lo = rr[:, :, i_lo]
                s_hi = rr[:, :, i_hi]

            # slice_c_lo: slice = slice(i_lo, i_lo+1)
            # slice_c_hi: slice = slice(i_hi, i_lo+1)

            # Mask of positive and negative velocity components
            mask_vel_pos: NumpyBoolArray = (vc > 0) # type: ignore
            mask_vel_neg: NumpyBoolArray = (vc < 0) # type: ignore
            # Mask of nonzero advection entries on live cells on this slice
            mask: NumpyBoolArray = (mask_vel_pos | mask_vel_neg) & mask_live

            # The number of entries with positive and negative velocity components
            nnz_slc: int = int(np.sum(mask))

            # Test whether velocity component is positive or negative. We need to use upwinding for stability.
            # When positive, we look to left; when negative, we look to right.


            # The main case is when we can look either to the left or the right depending on sign(u)
            if (0 < i) and (i < nc-1):
                # The row number of the cell interacting with cell r; in the direction dictated by upwinding
                s = s_lo * mask_vel_pos.astype(np.int32) + s_hi * mask_vel_neg.astype(np.int32)
                # Calculate the diagonal matrix entry for the advection operator
                diag_entry = -np.abs(vc) * (dt / h) # type: ignore
            # The case where we can only look to the right
            elif i == 0:
                s = s_hi
                diag_entry =  vc * (dt / h) # type: ignore
            # The case where we can only look to the left
            elif i == nc-1:
                s = s_lo
                diag_entry = -vc * (dt / h) # type: ignore

            # Set the diagonal entry for the advection operator
            slc = slice(nnz, nnz+nnz_slc)
            row[slc] = r[mask].flatten()
            col[slc] = r[mask].flatten()
            val[slc] = diag_entry[mask].flatten()
            nnz += nnz_slc
            # Set the off-diagonal entry for the advection operator
            slc = slice(nnz, nnz+nnz_slc)
            row[slc] = r[mask].flatten()
            col[slc] = s[mask].flatten()
            val[slc] = -diag_entry[mask].flatten()
            nnz += nnz_slc

    # Assemble the sparse matrix
    data: NumpyFloatArray = val[:nnz]
    coords: tuple[NumpyInt32Array, NumpyInt32Array] = (row[:nnz], col[:nnz])
    shape: tuple[int, int] = (N, N)
    dtype: np.dtype = np.float64 # type: ignore
    copy: bool = False
    A: coo_array = coo_array((data, coords), shape=shape, dtype=dtype, copy=copy)
    A.sum_duplicates()
    A.eliminate_zeros()
    return A.tocsr()

# *************************************************************************************************
def make_diffusion_op(geom: Geometry, diff: float, dt: float) -> csr_array:
    """
    Build an NxN advection matrix as a sparse matrix (COO format)
    INPUTS:
        geom - geometry object with grid information
        diff - diffusion coefficient
        dt - time step
    """
    # Unpack the dimensions from the geometry
    nx: int = geom.nx
    ny: int = geom.ny
    nz: int = geom.nz
    # Shape of the grid
    grid_shape: tuple[int, int, int] = (nx, ny, nz)
    # Unpack the grid row numbers
    rr: NumpyInt32Array = geom.rr
    # Unpack the grid spacing
    h: float = geom.h
    # Construct mask for live cells
    is_live: NumpyBoolArray = geom.is_fluid & ~geom.is_inlet
    # Number of rows in the SOC vector
    N: int = nx * ny * nz

    # The diffusion operator is given by
    # D = diff * Laplacian = diff * (d^2/dx^2 + d^2/dy^2 + d^2/dz^2)

    # The scale factor for diffusion operator entries in symmetric difference
    a: float = (diff * dt) / (h * h)
    # The scale factor for diffusion operator entries in one sided difference
    b: float = (diff * dt) / (h * h * h)

    # Upper bound on number of non-zero entries
    nnz_max: int = 10 * N
    # Counter for non-zero entries
    nnz: int = 0

    # An arbitrary entry in the advection matrix is at position (r, s) with value v
    # D[ row[p], col[p] ] = val[p]
    # Numpy array with row indices (r) of the non-zero entries
    row: NumpyInt32Array = np.zeros(nnz_max, dtype=np.int32)
    # Numpy array with column indices (s) of the non-zero entries
    col: NumpyInt32Array = np.zeros(nnz_max, dtype=np.int32)
    # Numpy array with values of the non-zero entries
    val: NumpyFloatArray = np.zeros(nnz_max, dtype=np.float64)
    # Current slice of the arrays
    slc: slice

    # Iterate over coordinates
    for c in range(3):
        # The number of slices in this coordinate
        nc: int = grid_shape[c]
        # Mask of liquid cells on this slice
        mask: NumpyBoolArray
        # The row index of the current slice; shape e.g. (ny, nz) when c=0
        r: NumpyInt32Array
        # The row index of the slice with x unshifted; middle of stencil
        s_mi: NumpyInt32Array
        # The row index of the slice with x shifted down 1; left of stencil
        s_lo: NumpyInt32Array
        # The row index of the slice with x shifted up 1; right of stencil
        s_hi: NumpyInt32Array
        # The row index of four columns used in one sided stencil at either edge
        s0: NumpyInt32Array
        s1: NumpyInt32Array
        s2: NumpyInt32Array
        s3: NumpyInt32Array

        # Iterate over slices of this coordinate
        for i in range(nc):
            # Value of i in the middle of the stencil; usually same as i except on the edges
            i_mi: int = int(np.clip(i, 1, nc-2))
            # Value of i for the left and right of the stencil
            i_lo: int = i_mi - 1
            i_hi: int = i_mi + 1
            # Calculate r, s_lo, s_mi, s_hi based on the coordinate
            # r is the unshifted cell. s_lo, s_mi, s_hi are cells at left, middle and right of three point stencil.
            if c == 0:
                mask = is_live[i, :, :]
                r    = rr[i   , :, :]
                s_lo = rr[i_lo, :, :]
                s_mi = rr[i_mi, :, :]
                s_hi = rr[i_hi, :, :]
                # Four columns for forward difference
                if i == 0:
                    s0 = rr[0, :, :]
                    s1 = rr[1, :, :]
                    s2 = rr[2, :, :]
                    s3 = rr[3, :, :]
                elif i == nc-1:
                    # Four columns for backward difference
                    s0 = rr[nc-1, :, :]
                    s1 = rr[nc-2, :, :]
                    s2 = rr[nc-3, :, :]
                    s3 = rr[nc-4, :, :]
            elif c == 1:
                mask = is_live[:, i, :]
                r    = rr[:, i   , :]
                s_lo = rr[:, i_lo, :]
                s_mi = rr[:, i_mi, :]
                s_hi = rr[:, i_hi, :]
                # Four columns for forward difference
                if i == 0:
                    s0 = rr[:, 0, :]
                    s1 = rr[:, 1, :]
                    s2 = rr[:, 2, :]
                    s3 = rr[:, 3, :]
                elif i == nc-1:
                    # Four columns for backward difference
                    s0 = rr[:, nc-1, :]
                    s1 = rr[:, nc-2, :]
                    s2 = rr[:, nc-3, :]
                    s3 = rr[:, nc-4, :]
            elif c == 2:
                mask = is_live[:, :, i]
                r    = rr[:, :, i   ]
                s_lo = rr[:, :, i_lo]
                s_mi = rr[:, :, i_mi]
                s_hi = rr[:, :, i_hi]
                # Four columns for forward difference
                if i == 0:
                    s0 = rr[:, :, 0]
                    s1 = rr[:, :, 1]
                    s2 = rr[:, :, 2]
                    s3 = rr[:, :, 3]
                elif i == nc-1:
                    # Four columns for backward difference
                    s0 = rr[:, :, nc-1]
                    s1 = rr[:, :, nc-2]
                    s2 = rr[:, :, nc-3]
                    s3 = rr[:, :, nc-4]

            # The number of entries on this slice (liquid cells only)
            nnz_slc: int = int(np.sum(mask))

            # The main case is when we can look either to the left or the right depending on sign(u)
            if (0 < i) and (i < nc-1):
                # The contribution to the diffusion operator from the low end of the stencil (i_lo with column s_lo)
                slc = slice(nnz, nnz+nnz_slc)
                row[slc] = r[mask].flatten()
                col[slc] = s_lo[mask].flatten()
                val[slc] = a
                nnz += nnz_slc
                # The contribution to the diffusion operator from the middle of the stencil (i_mi with column s_mi)
                slc = slice(nnz, nnz+nnz_slc)
                row[slc] = r[mask].flatten()
                col[slc] = s_mi[mask].flatten()
                val[slc] = -2.0 * a
                nnz += nnz_slc
                # The contribution to the diffusion operator from the high end of the stencil (i_hi with column s_hi)
                slc = slice(nnz, nnz+nnz_slc)
                row[slc] = r[mask].flatten()
                col[slc] = s_hi[mask].flatten()
                val[slc] = a
                nnz += nnz_slc

            # Apply the one sided stencil
            else:
                # The contribution to the diffusion operator f(x)
                slc = slice(nnz, nnz+nnz_slc)
                row[slc] = r[mask].flatten()
                col[slc] = s0[mask].flatten()
                val[slc] =  2.0 * b
                # The contribution to the diffusion operator f(x+h) or f(x-h)
                slc = slice(nnz, nnz+nnz_slc)
                row[slc] = r[mask].flatten()
                col[slc] = s1[mask].flatten()
                val[slc] = -5.0 * b
                # The contribution to the diffusion operator f(x+2h) or f(x-2h)
                slc = slice(nnz, nnz+nnz_slc)
                row[slc] = r[mask].flatten()
                col[slc] = s2[mask].flatten()
                val[slc] =  4.0 * b
                # The contribution to the diffusion operator f(x+3h) or f(x-3h)
                slc = slice(nnz, nnz+nnz_slc)
                row[slc] = r[mask].flatten()
                col[slc] = s3[mask].flatten()
                val[slc] = -1.0 * b

    # Assemble the sparse matrix
    data: NumpyFloatArray = val[:nnz]
    coords: tuple[NumpyInt32Array, NumpyInt32Array] = (row[:nnz], col[:nnz])
    shape_mat: tuple[int, int] = (N, N)
    dtype: np.dtype = np.float64 # type: ignore
    copy: bool = False
    D: coo_array = coo_array((data, coords), shape=shape_mat, dtype=dtype, copy=copy)
    D.sum_duplicates()
    D.eliminate_zeros()
    return D.tocsr()

# *************************************************************************************************
def adjust_op_wire(A: csr_array, geom: Geometry) -> csr_array:
    """
    Adjust an operator matrix so that cells on the wire are also updated.
    Idea: for every entry in the operator matrix of the type (r, s), where 
    r is a liquid cell and s is a wire cell, add a corresponding entry (s, r) that conserves mass.
    This includes an adjustment factor for the volume ratio.
    INPUTS:
        A - operator matrix; shape (N, N)
        geom - geometry object with grid information
    RETURNS:
        B - adjusted operator matrix; shape (N, N)
    """
    # Unpack masks from the geometry
    is_fluid: NumpyBoolArray = geom.is_fluid.flatten()
    is_wire: NumpyBoolArray = geom.is_wire.flatten()
    # Unpack the volumes from the geometry
    volume: NumpyFloatArray = geom.volume.flatten()
    # Convert the original operator to COO format
    A_coo = A.tocoo(copy=True)
    # Unpack the coordinates
    rr_A: NumpyInt32Array
    ss_A: NumpyInt32Array
    rr_A, ss_A = A_coo.coords
    # Unpack the matrix entries of A
    val_rs: NumpyFloatArray = A_coo.data
    # Extract the original entries where the row is a fluid and the column is a wire cell
    mask: NumpyBoolArray = is_fluid[rr_A] & is_wire[ss_A]
    # Assemble corresponding coordinates and values for these wire entries
    rr: NumpyInt32Array = rr_A[mask].copy()
    ss: NumpyInt32Array = ss_A[mask].copy()
    val_old: NumpyFloatArray = val_rs[mask].copy()
    # The corresponding volumes of these cells
    vol_r: NumpyFloatArray = volume[rr]
    vol_s: NumpyFloatArray = volume[ss]
    # Guard against divide-by-zero in wire cell volume.
    eps: float = np.finfo(np.float64).tiny
    mask_vol_ok: NumpyBoolArray = np.abs(vol_s) > eps
    n_zero_vol: int = int(np.size(vol_s) - np.count_nonzero(mask_vol_ok))
    if n_zero_vol > 0:
        print(f'Warning: adjust_op_wire() skipped {n_zero_vol} entries with near-zero wire volume.')

    ratio: NumpyFloatArray = np.zeros_like(vol_r)
    np.divide(vol_r, vol_s, out=ratio, where=mask_vol_ok)
    # The new entries on the diagonal
    val_diag: NumpyFloatArray = -val_old * ratio
    # The new entries off the diagonal
    val_off: NumpyFloatArray = -val_diag
    
    # Create a CSR operator for the new diagonal entries on the wire
    data_diag: NumpyFloatArray = val_diag
    coords_diag: tuple[NumpyInt32Array, NumpyInt32Array] = (ss, ss)
    shape: tuple[int, int] = A.shape
    dtype: np.dtype = np.float64
    Aw_diag: csr_array = coo_array((data_diag, coords_diag), shape=shape, dtype=dtype, copy=False).tocsr(copy=True)

    # Create a CSR operator for the new off diagonal entries on the wire
    data_off: NumpyFloatArray = val_off
    coords_off: tuple[NumpyInt32Array, NumpyInt32Array] = (ss, rr)
    shape: tuple[int, int] = A.shape
    dtype: np.dtype = np.float64
    Aw_off: coo_array = coo_array((data_off, coords_off), shape=shape, dtype=dtype, copy=False).tocsr(copy=True)

    # Return the sum of the original operator and the new entries
    B: csr_array = A + Aw_diag + Aw_off
    B.sum_duplicates()
    B.eliminate_zeros()
    return B

# *************************************************************************************************
def make_time_step_op(velocity: NumpyFloatArray, geom: Geometry, diff: float, dt: float, verbose: int) -> csr_array:
    """
    Build an NxN time step operator matrix as a sparse matrix (CSR format)
    INPUTS:
        velocity - velocity field; shape (nx, ny, nz, 3)
        geom - geometry object with grid information
        diff - diffusion coefficient
        dt - time step
        verbose - level of status messages
    RETURNS:
        T - time step operator matrix; shape (N, N)        
    """
    # Calculate the number of cells
    N: int = geom.nx * geom.ny * geom.nz
    # Build the advection operator
    A: csr_array = make_advection_op(velocity=velocity, geom=geom, dt=dt)
    # Build the diffusion operator
    D: csr_array = make_diffusion_op(geom=geom, diff=diff, dt=dt)
    # The sparse identity operator
    I: csr_array = csr_array(identity(N))

    # Build the total T operator as a CSR matrix
    T: csr_array = I + A + D
    T.sum_duplicates()
    T.eliminate_zeros()

    # Status message if requested
    if verbose > 0:
        print(f'make_time_step_op()')
        print(f'Step Size:   h = {geom.h:0.hf} cm')
        print(f'Step Time:  dt = {dt:0.6f} sec')
        print(f'Diffusivity: D = {diff:0.6e} cm^2/sec')
        print('\nOperator nonzero entries:')
        print(f'Advection: nnz = {A.nnz}')
        print(f'Diffusion: nnz = {D.nnz}')
        print(f'Time Step: nnz = {T.nnz}')

    # Operator detail if requested
    if verbose > 1:
        summarize_op(Op=A.tocsr(copy=True), geom=geom, name='Advection')
        summarize_op(Op=D.tocsr(copy=True), geom=geom, name='Diffusion')

    # Convert to CSR format and return
    return T.tocsr(copy=True)

# *************************************************************************************************
def summarize_op(Op: csr_array, geom: Geometry, name: str):
    """Summarize properties of an operator"""
    # Convert the masks to 1d
    is_fluid: NumpyBoolArray = geom.is_fluid.flatten()

    # Consolidate entries and eliminate zeros    
    Op.sum_duplicates()
    Op.eliminate_zeros()

    # Take sum over rows
    row_sum: NumpyFloatArray = np.array(Op.sum(axis=1)).flatten()
    # Filter to liquid cells
    r: NumpyFloatArray = row_sum[is_fluid]

    # Count of rows
    row_width = Op.max(axis=1) - Op.min(axis=1)
    r_count: int = int(row_width.count_nonzero())

    # Summary statistics by row sum
    r_min: float = float(np.min(r))
    r_max: float = float(np.max(r))
    r_mean: float = float(np.mean(r))
    r_std: float = float(np.std(r))
    r_pct_01: float = float(np.quantile(r, 0.01))
    r_pct_99: float = float(np.quantile(r, 0.99))

    # Summary statistics by entry
    a = Op.data
    a_count: int = len(a)
    a_mean: float = float(np.mean(a))
    a_std: float = float(np.std(a))
    a_min: float = float(np.min(a))
    a_max: float = float(np.max(a))
    a_pct_01: float = float(np.quantile(a, 0.01))
    a_pct_99: float = float(np.quantile(a, 0.99))

    # Summary statistics by abs(entry)
    b = np.abs(a)
    b_mean: float = float(np.mean(b))
    b_std: float = float(np.std(b))
    b_min: float = float(np.min(b))
    b_max: float = float(np.max(b))
    b_pct_01: float = float(np.quantile(b, 0.01))
    b_pct_99: float = float(np.quantile(b, 0.99))

    # Report results
    print(f'\n{name} operator summary over row sum and row entry:')
    print(f"stat    : row       : entry      : abs(entry)")
    print(f'count   : {r_count:9d} : {a_count:9d}')
    print(f'mean    : {r_mean  :9.2e} : {a_mean  :9.2e} : {b_mean  :9.2e}')
    print(f'std     : {r_std   :9.2e} : {a_std   :9.2e} : {b_std   :9.2e}')
    print(f'min     : {r_min   :9.2e} : {a_min   :9.2e} : {b_min   :9.2e}')
    print(f'max     : {r_max   :9.2e} : {a_max   :9.2e} : {b_max   :9.2e}')
    print(f'pct 01  : {r_pct_01:9.2e} : {a_pct_01:9.2e} : {b_pct_01:9.2e}')
    print(f'pct 99  : {r_pct_99:9.2e} : {a_pct_99:9.2e} : {b_pct_99:9.2e}')


# *************************************************************************************************
# Conservative finite-volume transport (added 2026-11-03)
#
# The legacy operators above do not conserve the charged species:
#   1. make_diffusion_op drops the one-sided stencil at every domain edge: its four writes share
#      one slice and nnz is never advanced, so the next slice overwrites them. A cell on a y or z
#      wall therefore has no diffusion along that axis, while its neighbor still exchanges with
#      it. Charged species that diffuse toward a wall disappear.
#   2. make_advection_op uses the upwind form u . grad(s) with the cell-centered velocity. That
#      velocity is not divergence-free on the grid, so the form creates or destroys charged
#      species wherever the SOC varies.
#   3. Both use one spacing h = dx for all three axes, but the grids are anisotropic.
#
# The functions below build the transport operator from fluxes through cell faces instead:
#   * the true spacing (hx, hy, hz) on every axis;
#   * no flux through the y and z channel walls, or, with periodic_y, one periodic y face;
#   * diffusion only (no advection) through faces between electrolyte and fiber cells;
#   * volumetric face fluxes that are projected to be exactly divergence-free on the electrolyte
#     cells, so the upwind scheme is conservative and also keeps a uniform SOC uniform.
# What one cell loses through a face, its neighbor gains, so at steady state the charged species
# released by the fibers equals the amount carried out through the outlet plus the amount that
# diffuses back into the inlet plane (see transport_budget).
# *************************************************************************************************

def grid_spacing(geom: Geometry) -> NumpyFloatArray:
    """Cell size (hx, hy, hz) in cm, from the domain extent and the grid shape."""
    return (np.asarray(geom.hi, dtype=np.float64) - np.asarray(geom.lo, dtype=np.float64)) \
        / np.array(geom.shape, dtype=np.float64)


def _pair_slices(ax: int) -> tuple[tuple[slice, ...], tuple[slice, ...], tuple[slice, ...]]:
    """Slices for the low cell, the high cell and the interior faces of each cell pair along ax."""
    lo = tuple(slice(0, -1) if a == ax else slice(None) for a in range(3))
    hi = tuple(slice(1, None) if a == ax else slice(None) for a in range(3))
    inner = tuple(slice(1, -1) if a == ax else slice(None) for a in range(3))
    return lo, hi, inner


def _wrap_slices(ax: int) -> tuple[tuple[slice, ...], tuple[slice, ...], tuple[slice, ...], tuple[slice, ...]]:
    """
    Slices for the periodic wrap pair along ax: the low cell is the last cell (index n-1), the high
    cell is the first (index 0), and the face between them is stored twice, at face index 0 and at
    face index n. All four slices keep a length-1 axis so the arrays line up.
    """
    lo = tuple(slice(-1, None) if a == ax else slice(None) for a in range(3))
    hi = tuple(slice(0, 1) if a == ax else slice(None) for a in range(3))
    face_first = tuple(slice(0, 1) if a == ax else slice(None) for a in range(3))
    face_last = tuple(slice(-1, None) if a == ax else slice(None) for a in range(3))
    return lo, hi, face_first, face_last


@dataclass
class FaceFlux:
    """Volumetric flux through every cell face, in cm^3/s, positive along +x, +y and +z.

    q[0] has shape (nx+1, ny, nz): q[0][i] is the face between cells i-1 and i, q[0][0] is the
    inlet boundary and q[0][nx] the outlet boundary. q[1] and q[2] are laid out the same way
    along y and z; their boundary faces are channel walls and carry no flux.

    With periodic_y, the y boundary faces q[1][:, 0] and q[1][:, ny] are the same physical face,
    between cell ny-1 and cell 0; both entries hold its flux, positive along +y, so _divergence
    needs no special case.
    """
    q: list[NumpyFloatArray]
    info: dict
    periodic_y: bool = False


def _divergence(q: list[NumpyFloatArray], shape: tuple[int, int, int]) -> NumpyFloatArray:
    """Net volumetric outflow of each cell, cm^3/s."""
    div = np.zeros(shape, dtype=np.float64)
    for ax in range(3):
        lo = tuple(slice(0, -1) if a == ax else slice(None) for a in range(3))
        hi = tuple(slice(1, None) if a == ax else slice(None) for a in range(3))
        div += q[ax][hi] - q[ax][lo]
    return div


def make_face_flux(velocity: NumpyFloatArray, geom: Geometry, project: bool = True,
                   rtol: float = 1.0E-10, verbose: int = 1, periodic_y: bool = False) -> FaceFlux:
    """
    Volumetric face fluxes for the finite-volume transport operator.
    INPUTS:
        velocity   - cell-centered velocity in cm/s; shape (nx, ny, nz, 3)
        geom       - geometry object with cell types
        project    - make the fluxes exactly divergence-free on the electrolyte cells (recommended)
        rtol       - relative residual for the projection's Poisson solve
        verbose    - print a summary when > 0
        periodic_y - treat the two y faces as one periodic face (the flow solve must be periodic
                     in y too); default False keeps them as walls with no flux
    The starting flux on a face between two electrolyte cells is the face area times the average
    of the two cell velocities. Faces that touch a fiber or a channel wall carry no flux. The
    inlet and outlet boundary faces take the velocity of the adjacent electrolyte cell.

    The projection is the discrete analog of incflo's MAC projection: it finds the
    divergence-free flux closest to the starting flux by solving a Poisson equation for a
    potential phi on the electrolyte cells, with phi = 0 at the outlet plane and no flux through
    walls and fiber faces. The inlet fluxes are kept as they are, so the total flow rate is exactly
    incflo's (the same number utils.params reports); fixing phi at the inlet as well would let
    the projection change the flow rate by a few percent. Electrolyte with no path to the outlet
    cannot carry a net flow, so its face fluxes (including any inlet faces) are set to zero.
    """
    shape: tuple[int, int, int] = geom.shape
    N: int = geom.N
    h: NumpyFloatArray = grid_spacing(geom)
    vol: float = float(np.prod(h))
    area: NumpyFloatArray = vol / h
    fl: NumpyBoolArray = geom.is_fluid
    t0: float = time.time()
    idx_t = np.int32 if N < 2**31 else np.int64

    # Starting fluxes: interior faces between two electrolyte cells, plus the inlet/outlet faces
    q: list[NumpyFloatArray] = []
    for ax in range(3):
        lo, hi, inner = _pair_slices(ax)
        qa = np.zeros(tuple(n + (1 if a == ax else 0) for a, n in enumerate(shape)))
        qa[inner] = area[ax] * 0.5 * (velocity[lo + (ax,)] + velocity[hi + (ax,)]) * (fl[lo] & fl[hi])
        q.append(qa)
    q[0][0] = area[0] * velocity[0, :, :, 0] * fl[0]
    q[0][-1] = area[0] * velocity[-1, :, :, 0] * fl[-1]
    # Periodic y: the face between cell ny-1 and cell 0, stored at both y boundary faces
    w_lo, w_hi, w_first, w_last = _wrap_slices(1)
    if periodic_y:
        q_wrap = area[1] * 0.5 * (velocity[w_lo + (1,)] + velocity[w_hi + (1,)]) * (fl[w_lo] & fl[w_hi])
        q[1][w_first] = q_wrap
        q[1][w_last] = q_wrap

    # Electrolyte cells connected (through electrolyte faces) to the outlet plane
    idx = np.arange(N, dtype=idx_t).reshape(shape)
    e_lo, e_hi = [], []
    for ax in range(3):
        lo, hi, _ = _pair_slices(ax)
        both = fl[lo] & fl[hi]
        e_lo.append(idx[lo][both])
        e_hi.append(idx[hi][both])
    if periodic_y:
        both = fl[w_lo] & fl[w_hi]
        e_lo.append(idx[w_lo][both])
        e_hi.append(idx[w_hi][both])
    e_lo_all = np.concatenate(e_lo)
    e_hi_all = np.concatenate(e_hi)
    adj = coo_array((np.ones(e_lo_all.size, dtype=np.int8), (e_lo_all, e_hi_all)), shape=(N, N))
    _, label = connected_components(adj.tocsr(), directed=False)
    label = label.reshape(shape)
    del adj, e_lo_all, e_hi_all
    is_open: NumpyBoolArray = fl & np.isin(label, np.unique(label[-1][fl[-1]]))
    del label
    n_closed: int = int(np.sum(fl & ~is_open))
    if n_closed > 0:
        for ax in range(3):
            lo, hi, inner = _pair_slices(ax)
            q[ax][inner] *= (is_open[lo] & is_open[hi])
        q[0][0] *= is_open[0]
        if periodic_y:
            open_wrap = is_open[w_lo] & is_open[w_hi]
            q[1][w_first] *= open_wrap
            q[1][w_last] *= open_wrap

    q_in_raw: float = float(np.sum(q[0][0]))
    q_out_raw: float = float(np.sum(q[0][-1]))
    div_raw: NumpyFloatArray = _divergence(q, shape)
    div_raw_rel: float = float(np.sum(np.abs(div_raw[is_open]))) / max(abs(q_in_raw), 1e-300)
    q_raw_norm: float = float(np.sqrt(sum(np.sum(qa * qa) for qa in q)))
    info: dict = dict(Q_in_raw=q_in_raw, Q_out_raw=q_out_raw, div_raw_rel=div_raw_rel,
                      n_closed_cells=n_closed, projected=project)

    if project:
        q_raw: list[NumpyFloatArray] = [qa.copy() for qa in q]
        # Unknown phi on every open electrolyte cell
        n_open: int = int(np.sum(is_open))
        unk = np.full(shape, -1, dtype=idx_t)
        unk[is_open] = np.arange(n_open, dtype=idx_t)
        rows, cols, vals = [], [], []
        diag = np.zeros(n_open, dtype=np.float64)
        for ax in range(3):
            lo, hi, _ = _pair_slices(ax)
            both = is_open[lo] & is_open[hi]
            a, b = unk[lo][both], unk[hi][both]
            beta: float = area[ax] / h[ax]
            rows += [a, b]
            cols += [b, a]
            vals += [np.full(a.size, -beta), np.full(a.size, -beta)]
            np.add.at(diag, a, beta)
            np.add.at(diag, b, beta)
        if periodic_y:
            both = is_open[w_lo] & is_open[w_hi]
            a, b = unk[w_lo][both], unk[w_hi][both]
            beta = area[1] / h[1]
            rows += [a, b]
            cols += [b, a]
            vals += [np.full(a.size, -beta), np.full(a.size, -beta)]
            np.add.at(diag, a, beta)
            np.add.at(diag, b, beta)
        # Dirichlet phi = 0 on the outlet boundary faces, half a cell from the cell center
        beta_b: float = area[0] / (0.5 * h[0])
        np.add.at(diag, unk[-1][is_open[-1]], beta_b)
        rows.append(np.arange(n_open, dtype=idx_t))
        cols.append(np.arange(n_open, dtype=idx_t))
        vals.append(diag)
        P = coo_array((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(n_open, n_open)).tocsr()
        del rows, cols, vals
        rhs: NumpyFloatArray = -div_raw[is_open]
        # Scale so the solve works with O(1) numbers
        scale: float = float(np.max(np.abs(rhs))) if np.any(rhs) else 1.0
        rhs = rhs / scale
        solver: str
        n_iter: int
        try:
            import pyamg
            ml = pyamg.smoothed_aggregation_solver(P, symmetry='symmetric')
            res: list = []
            phi = ml.solve(rhs, tol=rtol, accel='cg', maxiter=1000, residuals=res)
            solver, n_iter = 'pyamg', len(res) - 1
        except ImportError:
            count = [0]
            def _cb(_):
                count[0] += 1
            phi, _ = cg(P, rhs, rtol=rtol, maxiter=100000, M=diags(1.0 / P.diagonal()), callback=_cb)
            solver, n_iter = 'cg-jacobi', count[0]
        phi_rel_res: float = float(np.linalg.norm(P @ phi - rhs) / np.linalg.norm(rhs))
        if phi_rel_res > 1.0E-6:
            print(f'WARNING: projection Poisson solve stopped at relative residual {phi_rel_res:.1e}; '
                  'the face fluxes are not divergence-free and the budget will not close.')
        phi3 = np.zeros(shape, dtype=np.float64)
        phi3[is_open] = phi * scale
        del P, phi
        # Correct the fluxes: q = q* - beta * (phi_hi - phi_lo) on faces between open cells
        for ax in range(3):
            lo, hi, inner = _pair_slices(ax)
            q[ax][inner] -= (area[ax] / h[ax]) * (phi3[hi] - phi3[lo]) * (is_open[lo] & is_open[hi])
        q[0][-1] += beta_b * phi3[-1] * is_open[-1]
        if periodic_y:
            dq_wrap = (area[1] / h[1]) * (phi3[w_hi] - phi3[w_lo]) * (is_open[w_lo] & is_open[w_hi])
            q[1][w_first] -= dq_wrap
            q[1][w_last] -= dq_wrap
        div_new = _divergence(q, shape)
        dq_norm: float = float(np.sqrt(sum(np.sum(np.square(qa - qb)) for qa, qb in zip(q, q_raw))))
        del q_raw
        info.update(solver=solver, iterations=n_iter, poisson_rel_residual=phi_rel_res,
                    div_rel=float(np.sum(np.abs(div_new[is_open]))) / max(abs(float(np.sum(q[0][0]))), 1e-300),
                    flux_change_rel=dq_norm / max(q_raw_norm, 1e-300))

    info.update(Q_in=float(np.sum(q[0][0])), Q_out=float(np.sum(q[0][-1])),
                seconds=time.time() - t0, periodic_y=periodic_y)
    if periodic_y:
        # Net flow through the periodic y face: zero by symmetry for a fiber across the flow;
        # nonzero means a net sideways flow around the period
        info['Q_wrap'] = float(np.sum(q[1][w_first]))
    if verbose > 0:
        print('\nFace fluxes for the finite-volume transport operator:')
        print(f'spacing (hx, hy, hz) = ({h[0]*1e4:.4f}, {h[1]*1e4:.4f}, {h[2]*1e4:.4f}) um')
        if periodic_y:
            print(f'periodic in y: net flow through the periodic face = {info["Q_wrap"]*6e4:.6e} uL/min')
        print(f'enclosed electrolyte cells with no through-flow: {n_closed:d}')
        print(f'starting fluxes: Q_in = {q_in_raw*6e4:.6e} uL/min, Q_out = {q_out_raw*6e4:.6e} uL/min, '
              f'sum|div|/Q_in = {div_raw_rel:.3e}')
        if project:
            print(f'projected ({info["solver"]}, {info["iterations"]} iterations, Poisson residual '
                  f'{info["poisson_rel_residual"]:.1e}): Q_in = {info["Q_in"]*6e4:.6e} uL/min, '
                  f'Q_out = {info["Q_out"]*6e4:.6e} uL/min, sum|div|/Q_in = {info["div_rel"]:.3e}, '
                  f'relative flux change {info["flux_change_rel"]:.3e}')
        print(f'face fluxes built in {info["seconds"]:.1f} s')
    return FaceFlux(q=q, info=info, periodic_y=periodic_y)


def make_transport_rate_op(flux: FaceFlux, geom: Geometry, diff: float) -> csr_array:
    """
    Finite-volume advection-diffusion operator L in 1/s: d(SOC)/dt = L @ SOC.
    INPUTS:
        flux - face fluxes from make_face_flux
        geom - geometry object with cell types and the per-cell volume of wire cells
        diff - diffusion coefficient in cm^2/s
    Rows exist for live electrolyte cells (not the inlet plane, which holds the feed SOC) and for
    wire cells that share a face with a live electrolyte cell. A wire row is that cell's loss to
    the electrolyte divided by its own volume, so what a fiber releases is exactly what the
    electrolyte receives; nernst.py's rate law then sets the overpotential that resupplies it.
    Advection is upwind and conservative; the outlet face carries the outlet cell's own SOC
    (zero-gradient outflow). Diffusion has no flux through the y/z walls or the outlet face.
    If flux.periodic_y, the two y faces are one periodic face instead of walls: the last and first
    y cells exchange through it exactly as neighboring interior cells do.
    """
    shape: tuple[int, int, int] = geom.shape
    N: int = geom.N
    h: NumpyFloatArray = grid_spacing(geom)
    vol: float = float(np.prod(h))
    area: NumpyFloatArray = vol / h
    fl: NumpyBoolArray = geom.is_fluid
    wi: NumpyBoolArray = geom.is_wire
    live: NumpyBoolArray = fl & ~geom.is_inlet
    idx = np.arange(N, dtype=np.int32 if N < 2**31 else np.int64).reshape(shape)

    rows: list[NumpyInt32Array] = []
    cols: list[NumpyInt32Array] = []
    vals: list[NumpyFloatArray] = []

    def add(r, c, v) -> None:
        rows.append(r)
        cols.append(c)
        vals.append(np.broadcast_to(v, r.shape).astype(np.float64))

    def add_pair(lo, hi, qf, G) -> None:
        """Entries for every face between a low cell (slice lo) and a high cell (slice hi)."""
        qp = np.maximum(qf, 0.0)                    # flow from the low cell to the high cell
        qm = np.maximum(-qf, 0.0)                   # flow from the high cell to the low cell
        i_lo, i_hi = idx[lo], idx[hi]
        ff = fl[lo] & fl[hi]
        # Electrolyte-electrolyte faces: upwind advection plus diffusion
        m = ff & live[lo]
        add(i_lo[m], i_lo[m], -(G + qp[m]))
        add(i_lo[m], i_hi[m], G + qm[m])
        m = ff & live[hi]
        add(i_hi[m], i_hi[m], -(G + qm[m]))
        add(i_hi[m], i_lo[m], G + qp[m])
        # Electrolyte-fiber faces: diffusion only, both rows
        fw = (live[lo] & wi[hi]) | (wi[lo] & live[hi])
        add(i_lo[fw], i_lo[fw], -G)
        add(i_lo[fw], i_hi[fw], G)
        add(i_hi[fw], i_hi[fw], -G)
        add(i_hi[fw], i_lo[fw], G)

    for ax in range(3):
        lo, hi, inner = _pair_slices(ax)
        G: float = diff * area[ax] / h[ax]          # diffusive conductance, cm^3/s
        add_pair(lo, hi, flux.q[ax][inner], G)
    if flux.periodic_y:
        w_lo, w_hi, w_first, _ = _wrap_slices(1)
        add_pair(w_lo, w_hi, flux.q[1][w_first], diff * area[1] / h[1])

    # Outlet boundary faces: each outlet cell carries its own SOC out
    m = live[-1]
    add(idx[-1][m], idx[-1][m], -flux.q[0][-1][m])

    K = coo_array((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                  shape=(N, N)).tocsr()
    del rows, cols, vals
    K.sum_duplicates()
    # Divide each row by the volume of its cell: true volume for electrolyte, geom.volume for wire
    inv_vol = np.zeros(N, dtype=np.float64)
    inv_vol[live.ravel()] = 1.0 / vol
    w = wi.ravel()
    wv = np.asarray(geom.volume, dtype=np.float64).ravel()
    inv_vol[w & (wv > 0)] = 1.0 / wv[w & (wv > 0)]
    L = csr_array(diags(inv_vol) @ K)
    L.eliminate_zeros()
    return L


def calc_dt_op(L: csr_array, geom: Geometry, cfl: float) -> float:
    """
    Time step for explicit Euler with the operator L (1/s).
    The update s + dt * L s is a weighted average of neighbors (so bounded and monotone) when
    dt * |L_rr| <= 1 on every row. dt is cfl times that limit on the electrolyte rows, and no more
    than the limit on the wire rows, so a wire cell's transport update never overshoots below zero
    before nernst.py's rate law resets it.
    """
    d: NumpyFloatArray = -L.diagonal()
    d_fluid: float = float(np.max(d[geom.is_live.ravel()]))
    w = geom.is_wire.ravel()
    d_wire: float = float(np.max(d[w])) if np.any(w) else 0.0
    dt: float = cfl / d_fluid
    if d_wire > 0.0:
        dt = min(dt, 1.0 / d_wire)
    return dt


def transport_budget(soc: NumpyFloatArray, flux: FaceFlux, geom: Geometry, diff: float,
                     feed: float) -> dict:
    """
    Charged-species budget of a SOC field under the finite-volume operator, in cm^3/s of SOC
    (multiply by ne * F * c0 for mA). Charged species means SOC above the feed value.
        fiber  : released by the fibers into the electrolyte (diffusion through fiber faces)
        outlet : carried out through the outlet plane
        inlet  : lost back into the inlet plane, which is held at the feed SOC
        storage: fiber - outlet - inlet; zero at steady state
    """
    s = np.asarray(soc, dtype=np.float64).reshape(geom.shape)
    h = grid_spacing(geom)
    area = float(np.prod(h)) / h
    fl, wi = geom.is_fluid, geom.is_wire
    live = fl & ~geom.is_inlet
    fiber = 0.0
    pairs = [(_pair_slices(ax)[:2], ax) for ax in range(3)]
    if flux.periodic_y:
        pairs.append((_wrap_slices(1)[:2], 1))
    for (lo, hi), ax in pairs:
        G = diff * area[ax] / h[ax]
        m = live[lo] & wi[hi]
        fiber += G * float(np.sum(s[hi][m] - s[lo][m]))
        m = wi[lo] & live[hi]
        fiber += G * float(np.sum(s[lo][m] - s[hi][m]))
    outlet = float(np.sum(flux.q[0][-1] * (s[-1] - feed) * live[-1]))
    # Face between the inlet plane (i = 0) and the first live plane (i = 1)
    G = diff * area[0] / h[0]
    q01 = flux.q[0][1]
    m = fl[0] & live[1]
    inlet = float(np.sum((G * (s[1] - s[0]) + np.maximum(-q01, 0.0) * (s[1] - feed)
                          - np.maximum(q01, 0.0) * (s[0] - feed)) * m))
    return dict(fiber=fiber, outlet=outlet, inlet=inlet, storage=fiber - outlet - inlet)
