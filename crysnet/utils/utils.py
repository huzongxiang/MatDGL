# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 12:48:10 2021

@author: huzongxiang
"""

import numpy as np
from typing import List
from enum import Enum, unique
from pymatgen.core import Structure
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


@unique
class Features(Enum):
    atom = 1
    bond = 2
    image = 3
    state = 4
    pair_indices = 5
    lattice = 6
    cart_coords = 7
    
    def __int__(self):
        return self.value

    def __float__(self):
        return float(self.value)
    
    def __str__(self):
        return str(self.value)
    

def adjacent_matrix(num_atoms, pair_indices):
    """
    Parameters
    ----------
    num_atoms : TYPE
        DESCRIPTION.
    pair_indices : TYPE
        DESCRIPTION.

    Returns
    -------
    adj_matrix : TYPE
        DESCRIPTION.

    """
    adj_matrix = np.eye(num_atoms, dtype=np.float32)
    for pair_indice in pair_indices:
        begin_atom = pair_indice[0]
        end_atom = pair_indice[1]
        adj_matrix[begin_atom, end_atom] = adj_matrix[end_atom, begin_atom] = 1
    return adj_matrix


def get_valences(structure:Structure) -> np.ndarray:
    """
    Parameters
    ----------
    structure : Structure
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    BV = BVAnalyzer(symm_tol=0.1)
    try:
        sites_valences = BV.get_oxi_state_decorated_structure(structure)
    except:
        structure.add_oxidation_state_by_guess()
        sites_valences = structure
    atoms_valences = []
    for specie in sites_valences.species:
        atoms_valences.append(specie.oxi_state)
    return np.array(atoms_valences)


def get_space_group_number(structure: Structure) -> List:
    """
    Parameters
    ----------
    structure : structure
        DESCRIPTION.

    Returns
    -------
    List
        DESCRIPTION.

    """
    return SpacegroupAnalyzer(structure).get_space_group_number()


def get_space_group_info(structure: Structure) :
    """
    Parameters
    ----------
    structure : structure
        DESCRIPTION.

    Returns
    -------
    SpacegroupAnalyzer
        including symmetry info of structrue.

    """
    return SpacegroupAnalyzer(structure) 