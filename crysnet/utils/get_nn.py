# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 12:48:15 2021

@author: huzongxiang
"""


from typing import Tuple
import numpy as np
from pymatgen.optimization.neighbors import find_points_in_spheres


def get_nn_info(
    structure, cutoff: float = 3.2, numerical_tol: float = 1e-8
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    structure : Structure
        DESCRIPTION.
    cutoff : float, optional
        DESCRIPTION. The default is 5.0.
    numerical_tol : float, optional
        DESCRIPTION. The default is 1e-8.

    Returns
    -------
    TYPE
        np.ndarray.
    TYPE
        np.ndarray.
    TYPE
        np.ndarray.
    TYPE
        np.ndarray.

    """
    lattice_matrix = np.ascontiguousarray(np.array(structure.lattice.matrix), dtype=float)
    pbc = np.array([1, 1, 1], dtype=int)
    r = float(cutoff)
    cart_coords = np.ascontiguousarray(np.array(structure.cart_coords), dtype=float)
    center_indices, neighbor_indices, images, distances = find_points_in_spheres(
        cart_coords, cart_coords, r=r, pbc=pbc, lattice=lattice_matrix, tol=numerical_tol
    )
    center_indices = center_indices.astype(np.int32)
    neighbor_indices = neighbor_indices.astype(np.int32)
    images = images.astype(np.int32)
    distances = distances.astype(np.float32)
    
    return center_indices, neighbor_indices, images, distances