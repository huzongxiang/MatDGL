# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 15:17:09 2021

@author: huzongxiang
"""

import json
from tqdm import trange
import numpy as np
from typing import List, Dict, Any, Union
from mendeleev import element
from atom2vec import AtomSimilarity
from pymatgen.core import Structure, Element
from pymatgen.core.periodic_table import get_el_sp


class Atom2Vector:
    """
    Using principle of word2vec to embedding atom number to a vector.
    Default using e_above_hull to select structures in mp database.
    """
    def __init__(self, embedding_dim=16,
                 max_elements=8,
                 structures:List[Structure] = None):
        """
        Parameters
        ----------
        embedding_dim : TYPE, optional
            DESCRIPTION. The default is 20. Dimension of sematic vector of element.
        max_elements : TYPE, optional
            DESCRIPTION. The default is 10. The maximum number of elements in a structure.
        Returns
        -------
        None.

        """
        self.structures: List[Structure]
        self.structures = structures
        self.atom_similarity = self.cal_atom_similarity(self.structures, 
                                                     k_dim=embedding_dim, max_elements=max_elements)         


    @staticmethod 
    def cal_atom_similarity(structures:List, k_dim, max_elements):
        """
        Parameters
        ----------
        structures : List[Structure]
            DESCRIPTION. a list of pymatgen.core.Structure.
            You can transform poscar or cif file to pymatgen.core.Structure
            by using Structure.from_file(poscar or cif file).
        k_dim : TYPE
            DESCRIPTION.
        max_elements : TYPE
            DESCRIPTION.
        mpr : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        atom_similarity : TYPE
            DESCRIPTION.

        """
        atom_similarity = AtomSimilarity.from_structures(structures, k_dim, max_elements)
        return atom_similarity


    def get_atom_feature(self, element: Element) -> List[float]:
        """
        Parameters
        ----------
        element : pymatgen.core.Element
            DESCRIPTION.

        Returns
        -------
        List
            transform atom to a vector by atom2vector.
        """

        if isinstance(element, Element):
            atom_vector = self.atom_similarity.get_atom_vector(element)
            return atom_vector
        
        raise ValueError("%s not pymatgen.core.Element" % str(element))
        
        
    def get_atom_feature_table(self) -> Dict:
        """
        Returns
        -------
        Dict
            obtain all atom feature of periodic_table.
        """
        atom_vector_table = {}
        for Z in range(1,119):
            element = get_el_sp(Z)
            try:
                atom_vector_table[Z] = self.get_atom_feature(element)
            except:
                continue
        return atom_vector_table      


class Mendeleev_property:
    def __init__(self):
        pass
    
    @staticmethod
    def get_mendeleev_properties():
        """
        Returns
        -------
        dict
            DESCRIPTION.
            Get properties of elements in mendeleev table.
        """
        dicts = {'s':0, 'p':1, 'd':2, 'f':3}
        
        nvalence = {}
        unpaired_electrons = {}
        electronegativity = {}
        angle_quantum_num = {}
        ionenergies = {}
        dipole_polarizability = {}
        covalent_radius = {}
        atomic_weight = {}
        
        for i in trange(1,119):
            key = str(i)
            nvalence[key] = element(i).nvalence()
            unpaired_electrons[key] = element(i).ec.unpaired_electrons()
            if i <= 116:
                electronegativity[key] = element(i).electronegativity('pauling') or element(i-1).electronegativity('pauling') 
            else:
                electronegativity[key] = element(116).electronegativity('pauling')
            angle_quantum_num[key] = dicts[element(i).block]
            if i <= 110:
                ionenergies[key] = list(element(i).ionenergies.values())[0]
            else:
                ionenergies[key] = list(element(110).ionenergies.values())[0]
            dipole_polarizability[key] = element(i).dipole_polarizability
            covalent_radius[key] = element(i).covalent_radius
            atomic_weight[key] = element(i).atomic_weight
            
        return {"nvalence": nvalence,
                "unpaired_electrons": unpaired_electrons,
                "electronegativity": electronegativity, 
                "angle_quantum_num": angle_quantum_num,
                "ionenergies": ionenergies,
                "dipole_polarizability": dipole_polarizability,
                "covalent_radius":covalent_radius,
                "atomic_weight": atomic_weight}
    
    
    def save_mendeleev_properties(mendeleev_properties):
        with open('mendeleev.json','w') as f:
            json.dump(mendeleev_properties, f)


class Converter:
    
    def convert(d: Any) -> Any:
        """
        Parameters
        ----------
        d : Any
            DESCRIPTION.

        Returns
        -------
        None.

        """
        pass


class GaussianDistance(Converter):
    """
    Expand distance with Gaussian basis sit at centers and with width 0.5.
    """
    def __init__(self, n=64, width=0.5):
        """
        Parameters
        ----------
        centers : np.ndarray, optional
            DESCRIPTION. The default is np.linspace(0, 5, 100).
        width : TYPE, optional
            DESCRIPTION. The default is 0.5.

        Returns
        -------
        None.

        """
        self.centers = np.linspace(0, 1, n)
        self.width = width


    def convert(self, distances: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        distances : np.ndarray
            expand distance vector distances with given parameters.

        Returns
        -------
        TYPE
            (matrix) N*M matrix with N the length of distances and M the length of centers.

        """
        distances = np.array(distances)
        return np.exp(-((distances[:, None] - self.centers[None, :])**2)/self.width**2)        


class MultiPropertyFeatures(Converter):
    """
    Get multi-property bond features.
    """
    def __init__(self, mendeleev_properties):
        self.properties = mendeleev_properties
        
    def convert(self, bond_indinces) -> np.ndarray:
        """
        Parameters
        ----------
        bond_indinces : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.
            
        """
        bonds_properties = []
        for bond_indince in bond_indinces:
            start, end = str(bond_indince[0]), str(bond_indince[1])
            ven = self.properties["nvalence"][start]/self.properties["nvalence"][end] # valence eletron
            if start != end:
                upe = self.properties["unpaired_electrons"][start] + self.properties["unpaired_electrons"][end] # upair eletron
                aqn = self.properties["angle_quantum_num"][start] + self.properties["angle_quantum_num"][end] # angle quantum number
            else:
                upe = self.properties["unpaired_electrons"][start] # upair eletron
                aqn = self.properties["angle_quantum_num"][start] # angle quantum number
            eng = self.properties["electronegativity"][start]/self.properties["electronegativity"][end] # electronegativity
            fie = self.properties["ionenergies"][start]/self.properties["ionenergies"][end] # the first ionenergies
            plb = self.properties["dipole_polarizability"][start]/self.properties["dipole_polarizability"][end] # dipole_polarizability
            cvr = self.properties["covalent_radius"][start]/self.properties["covalent_radius"][end] # covalent_radius

            bonds_properties.append([ven, upe, aqn, eng, fie, plb, cvr])
        return np.array(bonds_properties, dtype=np.float32)
    
            
class Embedding_edges:
    """
    Costomed bond feature mbedding using different bond properties.
    """
    def __init__(self, converter: Union[str, Converter]):
        self.converter = converter
        
    def embedding(self, d: Any) -> Any:
        """
        Parameters
        ----------
        d : Any
            DESCRIPTION.

        Returns
        -------
        Any
            DESCRIPTION.

        """
        return self.converter.convert(d)