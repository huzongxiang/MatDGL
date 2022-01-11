# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 14:56:51 2021

@author: huzongxiang
"""


import os
import json
import requests
from tqdm import tqdm
import numpy as np
from pathlib import Path
from typing import List, Dict, Union
from pymatgen.ext.matproj import MPRester
from pymatgen.core.structure import Structure


np.random.seed(42)


class Dataset:
    """
    Accessing materialsproject.org by api.
    """
    def __init__(self, task_type=None, data_path=None, api_key=None):
        if task_type not in ['my_regression', 'my_classification', 'my_multiclassification', 'metal', 'band_gap', 'dos_fermi', 'formation_energy', 'formation_energy_all', 'formation_energy', 'e_above_hull', 'topology', 'topology_multi', 'GVRH', 'KVRH', 'possion_ratio', 'regression', 'classification', 'multiclassification']:
            raise ValueError('Invalid task type! Input task_type should be my_regression, my_classification, my_multiclassification, regression, classification, multiclassification, metal, band_gap, dos_fermi, formation_energy, e_above_hull, formation_energy_all, formation_energy, topology, GVRH, KVRH, possion_ratio.')
        
        self.task_type = task_type
        self.multiclassification = False
        
        if task_type == 'band_gap':
            self.dataset_file = Path(data_path/"datas"/"dataset_gap.json")
            self.regression = True
        elif task_type == 'dos_fermi':
            self.dataset_file = Path(data_path/"datas"/"dataset_dos.json")
            self.regression = True
        elif task_type == 'formation_energy':
            self.dataset_file = Path(data_path/"datas"/"dataset_fe.json")
            self.regression = True
        elif task_type == 'formation_energy_all':
            self.dataset_file = Path(data_path/"datas"/"dataset_fe_all.json")
            self.regression = True
        elif task_type == 'e_above_hull':
            self.dataset_file = Path(data_path/"datas"/"dataset_ehull.json")
            self.regression = True
        elif task_type == 'metal':
            self.dataset_file = Path(data_path/"datas"/"dataset_metal.json")
            self.regression = False
        elif task_type == 'topology':
            self.dataset_file = Path(data_path/"datas"/"dataset_tp.json")
            self.regression = False
        elif task_type == 'GVRH':
            self.dataset_file = Path(data_path/"datas"/"dataset_GVRH.json")
            self.regression = True
        elif task_type == 'KVRH':
            self.dataset_file = Path(data_path/"datas"/"dataset_KVRH.json")
            self.regression = True
        elif task_type == 'possion_ratio':
            self.dataset_file = Path(data_path/"datas"/"dataset_pr.json")
            self.regression = True
        elif task_type == 'regression':
            self.dataset_file = Path(data_path/"datas"/"dataset_regression.json")
            self.regression = True
        elif task_type == 'classification':
            self.dataset_file = Path(data_path/"datas"/"dataset_classification.json")
            self.regression = False
        elif task_type == 'topology_multi':
            self.dataset_file = Path(data_path/"datas"/"dataset_tp_multi.json")
            self.regression = False
            self.multiclassification = True
        elif task_type == 'multiclassification':
            self.dataset_file = Path(data_path/"datas"/"dataset_multiclassification.json")
            self.regression = False
            self.multiclassification = True
        elif task_type == 'my_regression':
            self.dataset_file = Path(data_path/"datas"/"dataset_myr.json")
            self.regression = True
        elif task_type == 'my_classificaiton':
            self.dataset_file = Path(data_path/"datas"/"dataset_myc.json")
            self.regression = False
        elif task_type == 'my_multiclassification':
            self.dataset_file = Path(data_path/"datas"/"dataset_mym.json")
            self.regression = False
            self.multiclassification = True
        
        self.structures = None
        self.labels = None

        if self.dataset_file.exists():    
            with open(self.dataset_file, 'r') as f:
                entries = json.load(f)
            if isinstance(entries[0], list):
                self.structures = [Structure.from_dict(s[0]) for s in entries]
                self.labels = [s[1] for s in entries]
                self.dataset = zip(self.structures, self.labels)
                self.permuted_indices = self._permute_indices(self.labels)
                self.datasize = len(self.labels)
            else:
                self.structures = [Structure.from_dict(s) for s in entries]
                self.labels = None
                self.dataset = self.structures
                self.permuted_indices = self._permute_indices(self.structures)
                self.datasize = len(self.structures)

        if api_key is not None:
            self.mpr = MPRester(api_key)

    
    def prepare_datas(self,
                      criteria = {
                              "e_above_hull": {"$lte": 0.}
                                  },
                      properties = [
                              "structure",
                              # "final_energy",
                              "formation_energy_per_atom",
                                    ],
                      ):
        """
        Parameters
        ----------
        criteria : TYPE, optional
            criteria (str/dict) : pymatgen
            for example: {"formation_energy_per_atom": {"$exists": True}}
        properties : TYPE, optional
            DESCRIPTION. The default is ["structure","final_energy",].
            Properties to request for as a list. 
            For example, ["formula", "formation_energy_per_atom"] returns the formula and formation energy per atom.
        structures: TYPE, optional
            DESCRIPTION. The default is None. If you do not want to get structure dataset from materialsproject.org by MP API MPRester.
            You have prepared your dataset as form with List[your structures in pymatgen.core.Structure]
            to define the semantic space and get atom embedding vector.
        Returns
        -------
        None.

        """
        entries = self.mpr.query(criteria=criteria, properties=properties, mp_decode=True)
        structures = [s['structure'].as_dict() for s in entries]
        labels = [s[properties[1]] for s in entries]
        
        return structures, labels


    def save_datasets(self, structures: List, labels: Union[List, None]=None):
        """
        Saving structrues and label to json file to avoid re-download entries from materialsproject.org.
        """
        if isinstance(structures[0], Structure):
            structures_dict=[]
            for s in tqdm(structures, desc="Converting Structure to json dict"):
                structures_dict.append(s.as_dict())
        else:
            structures_dict = structures
        dataset = []
        if labels:
            for structure_label in zip(structures_dict, labels):
                dataset.append(structure_label)
        else:
            for structure in structures_dict:
                dataset.append(structure)            
        with open(self.dataset_file,'w') as f:
            json.dump(dataset, f)
            

    def _permute_indices(self, x):
        permuted_indices = np.random.permutation(np.arange(len(x)))
        return permuted_indices
    
    
    def permute_indices(self, x: int):
        if isinstance(x, int):
            permuted_indices = np.random.permutation(np.arange(x))
        return permuted_indices


    def shuffle_set(self, structures: List=None, labels: List=None) -> List:
        permuted_indices = np.random.permutation(np.arange(len(labels)))

        structures_shuffle = []
        labels_shuffle = []

        if labels is None:
            for index in permuted_indices:
                structures_shuffle.append(structures[index])
            return structures_shuffle
        for index in permuted_indices:
                structures_shuffle.append(structures[index])
                labels_shuffle.append(labels[index])
        return structures_shuffle, labels_shuffle


    def prepare_train_set(self, structures: List, labels: List=None, permutation=None) -> List:
        if structures:
            self.structures = structures
        if labels:
            self.labels = labels
        if permutation is not None:
           self.permuted_indices = permutation 

        train_index = self.permuted_indices[: int(len(self.permuted_indices) * 0.7)]
        x_train = []
        y_train = []
        for index in train_index:
            x_train.append(self.structures[index])
            y_train.append(self.labels[index])
        return x_train, y_train


    def prepare_validate_set(self, structures: List, labels: List=None, permutation=None) -> List:
        if structures:
            self.structures = structures
        if labels:
            self.labels = labels
        if permutation is not None:
           self.permuted_indices = permutation 

        valid_index = self.permuted_indices[int(len(self.permuted_indices) * 0.7) : int(len(self.permuted_indices) * 0.9)]
        x_valid = []
        y_valid = []
        for index in valid_index:
            x_valid.append(self.structures[index])
            y_valid.append(self.labels[index])
        return x_valid, y_valid


    def prepare_test_set(self, structures: List, labels: List=None, permutation=None) -> List:
        if structures:
            self.structures = structures
        if labels:
            self.labels = labels
        if permutation is not None:
           self.permuted_indices = permutation 

        # test_index = self.permuted_indices
        test_index = self.permuted_indices[int(len(self.permuted_indices) * 0.9) :]
        x_test = []
        y_test = []
        for index in test_index:
            x_test.append(self.structures[index])
            y_test.append(self.labels[index])
        return x_test, y_test


    def prepare_x(self, structures: List) -> List:
        if structures:
            self.structures = structures
        else:
            raise ValueError('strucutres should not be None.')
        return self.structure
    
    
    def prepare_y(self, labels: List=None) -> List:
        self.labels = labels
        return self.labels


class TopologicalDataset(Dataset):
    def __init__(self, token=None, api_key=None):
        self.mpr = MPRester(api_key)
        self.hostname = 'materiae.iphy.ac.cn/'
        if token is not None:
            self.token = token
        else:
            raise ValueError('token should not be None, get your token from "materiae.iphy.ac.cn"')
        
        
    def query_Materiae(self) -> List:
        """
        Returns
        -------
        Dict
            dict of datas from Materiae.

        """
        hostname = self.hostname
        token = self.token
        url = "http://%s/api/materials" % hostname
        headers = {'Authorization': 'Bearer %s' % token}    
        response = requests.get(url=url, headers=headers)
        materials = response.json()['materials']
        return materials
    

    def topological_classfication(self):
        """
        Returns
        -------
        strurtures : List of pymatgen.core.struture.Structure
            strcutures used for graph networks.
        labels : List of topological classfications.
            classification labels for supervised graph networks.

        """
        hostname = self.hostname
        token = self.token
        url = "http://%s/api/materials" % hostname
        headers = {'Authorization': 'Bearer %s' % token}    
        response = requests.get(url=url, headers=headers)
        materials = response.json()['materials']
        
        strurtures = []
        labels = []
        for material in tqdm(materials):
            url = "http://%s/api/materials/%s" % (hostname, material['id'])
            headers = {'Authorization': 'Bearer %s' % token}
            response = requests.get(url=url, headers=headers)
            mp_id = response.json()['mp_id']
            # nsoc_topo_class = response.json()['nsoc_topo_class']
            soc_topo_class = response.json()['soc_topo_class']
            is_topo = 0
            if soc_topo_class != 'Triv_Ins' and soc_topo_class != '':
                is_topo = 1
            structure = self.mpr.get_structure_by_material_id(mp_id, 
                                                              final=True, 
                                                              conventional_unit_cell=False)
            strurtures.append(structure.as_dict())
            labels.append(is_topo)
            print(structure.formula,' soc_topo_class: ', soc_topo_class,' is_topo: ', is_topo)
        return strurtures, labels


    def topological_multiclassfication(self):
        """
        Returns
        -------
        strurtures : List of pymatgen.core.struture.Structure
            strcutures used for graph networks.
        labels : List of topological classfications.
            classification labels for supervised graph networks.

        """
        hostname = self.hostname
        token = self.token
        url = "http://%s/api/materials" % hostname
        headers = {'Authorization': 'Bearer %s' % token}    
        response = requests.get(url=url, headers=headers)
        materials = response.json()['materials']
        
        strurtures = []
        labels = []
        for material in tqdm(materials):
            url = "http://%s/api/materials/%s" % (hostname, material['id'])
            headers = {'Authorization': 'Bearer %s' % token}
            response = requests.get(url=url, headers=headers)
            mp_id = response.json()['mp_id']
            # nsoc_topo_class = response.json()['nsoc_topo_class']
            soc_topo_class = response.json()['soc_topo_class']
            topo = 0
            if soc_topo_class == 'Triv_Ins' or soc_topo_class == '':
                topo = 0
            elif soc_topo_class == 'HSL_SM':
                topo = 1
            elif soc_topo_class == 'HSP_SM':
                topo = 2
            elif soc_topo_class == 'TCI':
                topo = 3
            elif soc_topo_class == 'TI':
                topo = 4
            structure = self.mpr.get_structure_by_material_id(mp_id, 
                                                          final=True, 
                                                          conventional_unit_cell=False)
            strurtures.append(structure.as_dict())
            labels.append(topo)
            print(structure.formula,' soc_topo_class: ', soc_topo_class,' is_topo: ', topo)
        return strurtures, labels


    def get_properties_Materiae(self, p: str) -> Dict:
        """
        Get properties from Materiae dataset.
        --properties:
             'id',
             'formula',
             'elements',
             'elements_num',
             'mp_id',
             'icsd_ids',
             'nelec',
             'nsites',
             'spacegroup.number',
             'spacegroup.lattice_type',
             'nsoc_efermi',
             'nsoc_dos',
             'nsoc_fermi_dos',
             'nsoc_dos_gap',
             'nsoc_band',
             'nsoc_topo_class',
             'nsoc_sym_ind',
             'nsoc_ind_group',
             'soc_efermi',
             'soc_dos',
             'soc_fermi_dos',
             'soc_dos_gap',
             'soc_band',
             'soc_topo_class',
             'soc_sym_ind',
             'soc_ind_group'
        Returns
        -------
        None.

        """
        hostname = self.hostname
        token = self.token
        url = "http://%s/api/materials" % hostname
        headers = {'Authorization': 'Bearer %s' % token}    
        response = requests.get(url=url, headers=headers)
        materials = response.json()['materials']
        
        datas = {}
        for material in materials:
            url = "http://%s/api/materials/%s" % (self.hostname, material['id'])
            headers = {'Authorization': 'Bearer %s' % token}
            response = requests.get(url=url, headers=headers) 
            mp_id = response.json()['mp_id']
            datas[mp_id] = response.json()[p]
        
        return datas
    
        
    def get_topology_info(self, params):     
        #params = {'nsoc_topo_class': 'GM_SM'}
        params = {'fields': params}
        hostname = self.hostname
        token = self.token
        url = "http://%s/api/materials" % hostname
        headers = {'Authorization': 'Bearer %s' % token}    
        response = requests.get(url=url, params=params, headers=headers)
        materials = response.json()['materials']
        return materials
        

    def download_Materiae(self):
        hostname = self.hostname
        token = self.token
        url = "http://%s/api/materials" % hostname
        headers = {'Authorization': 'Bearer %s' % token}    
        response = requests.get(url=url, headers=headers)
        materials = response.json()['materials']
        datas = {}
        for material in materials:
            url = "http://%s/api/materials/%s" % (self.hostname, material['id'])
            headers = {'Authorization': 'Bearer %s' % token}
            response = requests.get(url=url, headers=headers) 
            
            mp_id = response.json()['mp_id']
            datas[mp_id] = response.json()
