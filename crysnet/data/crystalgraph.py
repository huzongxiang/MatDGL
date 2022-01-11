# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 11:39:28 2021

@author: huzongxiang
"""

import time
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path
from operator import itemgetter
from multiprocessing import Pool
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from crysnet.utils import Features
from typing import Union, Dict, List, Set
from .embedding import Mendeleev_property, GaussianDistance, MultiPropertyFeatures, Embedding_edges
from pymatgen.core import Structure
from pymatgen.analysis.local_env import NearNeighbors, VoronoiNN
from crysnet.utils.get_nn import get_nn_info
from crysnet.utils import get_space_group_number


ModulePath = Path(__file__).parent.absolute()


class LabelledCrystalGraphBase():
    
    def __init__(self, strategy: Union[None, NearNeighbors]=VoronoiNN(cutoff=18.0)
                 ):
        """
        Parameters
        ----------
        strategy : Union[str, NearNeighbors], optional
            DESCRIPTION. The default is 'VoronoiNN'.
        atom2vector : Union[None, Atom2Vector], optional
            DESCRIPTION. The default is None. if atom2vector is None, program is going to download dataset
            from materilsproject.org to construct Semantic space of atoms, it will cost a lot of time, 
            so recommand set atom2vector=Atom2Vector() at first.

        Raises
        ------
        RuntimeError
            DESCRIPTION.

        Returns
        -------
        None.

        """
    
        if isinstance(strategy, NearNeighbors):
            self.strategy = strategy
        
        self.properties = None
        with open(Path(ModulePath/"mendeleev.json"),'r') as f:
            self.properties = json.load(f)
        if self.properties is None:
            self.properties = Mendeleev_property.get_mendeleev_properties()
        
        
    def get_graph(self, structure: Structure) -> Dict:
        """
        Parameters
        ----------
        structure : pymatgen.core.structure.Structure
            Feed with pymatgen.core.structure.Structure and produce the graph.

        Raises
        ------
        RuntimeError
            DESCRIPTION.

        Returns
        -------
        Dict
        Labelled graph
            state_attributes is global attributes, here is symmetry,
            oxide_states in order of atoms,
            atoms are nodes of graph,
            bond are distance between nodes,
            image are direction of bond along abc,
            pair_indices are nodes indices of edges, including self-cycle edge,
            lattice are abc of structure,
            cart_coords are Descartes coordinations.

        """
        lattice = np.array(structure.as_dict()['lattice']['matrix'])
        cart_coords = structure.cart_coords
        space_group_number = get_space_group_number(structure)
        state_attributes = np.array([space_group_number],dtype="int32")
        
        node1 = []
        node2 = []
        bonds = []
        images = []
        for atom, neighbors in enumerate(self.strategy.get_all_nn_info(structure)):
            node1.extend([atom] * (len(neighbors) + 1))
            node2.append(atom)
            bonds.append(0.0)
            images.append((0,0,0))
            for neighbor in neighbors:
                node2.append(neighbor["site_index"])
                bonds.append(neighbor["weight"])
                images.append(neighbor['image'])
        atoms = self.get_Z_number(structure)
        pair_indices = list(zip(node1,node2))
        if np.size(np.unique(node1)) < len(atoms):
            raise RuntimeError("Isolated atoms found in the structure")

        return {Features.atom: atoms, Features.bond: bonds, Features.state: state_attributes,
                Features.pair_indices: pair_indices, Features.image: images,
                Features.lattice: lattice, Features.cart_coords: cart_coords}

    
    @staticmethod
    def get_Z_number(structure: Structure) -> List:
        """
        Parameters
        ----------
        structure : Structure
            Get atomic number from pymatgen.core.struture.Structure.
            structure.atomic_number can also get it.
        Returns
        -------
        List
            DESCRIPTION.
        """
        return np.array([i.specie.Z for i in structure], dtype="int32").tolist()  
 

    def _local_coordinates(self, graph: Dict) -> List: 
        """
        Parameters
        ----------
        graph : Dict
            DESCRIPTION.

        Returns
        -------
        TYPE
            a seires of polar coordinates used for building local coordinate.

        """
        pair_indices = graph[Features.pair_indices]
        images = graph[Features.image]
        lattice = graph[Features.lattice]
        a, b, c = lattice[0], lattice[1], lattice[2]
        cart_coords = graph[Features.cart_coords]
        
        local_env = []
        for idx, pair_indice in enumerate(pair_indices):
            image = images[idx]
            node_send, node_recive = pair_indice[1], pair_indice[0]
            polar = cart_coords[node_recive] - cart_coords[node_send] - \
                    a*image[0] - b*image[1] - c*image[2]
            vetical = np.array([polar[1], -polar[0], 0.0])
            local_env.append(np.array([polar[0], polar[1], polar[2], vetical[0], vetical[1], vetical[2]]))
            
        return local_env

    
    def graph_to_input(self, graph: Dict) -> List[np.ndarray]:
        """
        Parameters
        ----------
        graph : Dict
            DESCRIPTION.

        Returns
        -------
        list
            DESCRIPTION.

        """
        atom_num_pairs = [[graph[Features.atom][pair[0]], graph[Features.atom][pair[1]]] for pair in graph[Features.pair_indices]]
        distance_features = Embedding_edges(converter=GaussianDistance()).embedding(graph[Features.bond])
        multi_properties = Embedding_edges(converter=MultiPropertyFeatures(self.properties)).embedding(atom_num_pairs)
        bond_features = np.concatenate([distance_features, multi_properties], axis=1)

        local_env = self._local_coordinates(graph)
        
        return [
            np.array(graph[Features.atom], dtype=np.int32),
            np.array(bond_features),
            np.array(graph[Features.state], dtype=np.int32),
            np.array(graph[Features.pair_indices], dtype=np.int32),
            np.array(local_env),
            ]


    def structure_to_input(self, structure: Structure) -> List:
        """
        Parameters
        ----------
        structure : pymatgen.core.structure.Structure
            DESCRIPTION.

        Returns
        -------
        List
            DESCRIPTION.

        """
        graph = self.get_graph(structure)
        return self.graph_to_input(graph)

    
    def ragged_inputs_from_strcutre_list(self, structure_list: List) -> Set:
        """
        Parameters
        ----------
        structure_list : List
            a list of pymatgen.core.Structure
            In order to keep with Semantic space of atom2vector, MPRester should be used
            to get structures from materialsproject.

        Returns
        -------
        Set
            DESCRIPTION.

        """
        # Initialize graphs
        atom_features_list = []
        bond_features_list = []
        state_attrs_list = []
        pair_indices_list = []
        local_env_list = []
        for structure in tqdm(structure_list, desc="Generating labelled multi-property graphs"):
            atom_features, bond_features, state_attrs, pair_indices, local_env = self.structure_to_input(structure)
            atom_features_list.append(atom_features)
            bond_features_list.append(bond_features)
            state_attrs_list.append(state_attrs)
            pair_indices_list.append(pair_indices)
            local_env_list.append(local_env)
  
        return (
            tf.ragged.constant(atom_features_list, dtype=tf.int32),
            tf.ragged.constant(bond_features_list, dtype=tf.float32),
            tf.ragged.constant(local_env_list, dtype=tf.float32),
            tf.ragged.constant(state_attrs_list, dtype=tf.int32),
            tf.ragged.constant(pair_indices_list, dtype=tf.int64),
            )


    def inputs_from_strcutre_list(self, structure_list: List) -> Set:
        """
        Parameters
        ----------
        structure_list : List
            a list of pymatgen.core.Structure
            In order to keep with Semantic space of atom2vector, MPRester should be used
            to get structures from materialsproject.

        Returns
        -------
        Set
            DESCRIPTION.

        """
        # Initialize graphs
        atom_features_list = []
        bond_features_list = []
        state_attrs_list = []
        pair_indices_list = []
        local_env_list = []
        for structure in tqdm(structure_list, desc="Generating labelled directionial spherical harmonic graphs"):
            atom_features, bond_features, state_attrs, pair_indices, local_env = self.structure_to_input(structure)
            atom_features_list.append(atom_features)
            bond_features_list.append(bond_features)
            state_attrs_list.append(state_attrs)
            pair_indices_list.append(pair_indices)
            local_env_list.append(local_env)
  
        return (
            atom_features_list,
            bond_features_list,
            local_env_list,
            state_attrs_list,
            pair_indices_list,
            )


    def graphs_from_strcutre_list(self, structure_list: List) -> List:
        """
        Parameters
        ----------
        structure_list : List
            a list of pymatgen.core.Structure
            In order to keep with Semantic space of atom2vector, MPRester should be used
            to get structures from materialsproject.

        Returns
        -------
        List
            DESCRIPTION.

        """
        # Initialize graphs
        graphs = []
        for structure in tqdm(structure_list, desc="Generating labelled directionial spherical harmonic graphs"):
            graphs.append(self.structure_to_input(structure))
  
        return graphs


class LabelledCrystalGraph(LabelledCrystalGraphBase):
    def __init__(self, cutoff=3.0, mendeleev=False):
        self.cutoff = cutoff
        self.mendeleev = mendeleev
        if self.mendeleev:
            with open(Path(ModulePath/"mendeleev.json"),'r') as f:
                self.properties = json.load(f)


    def graph_to_input(self, graph: Dict) -> List[np.ndarray]:
        """
        Parameters
        ----------
        graph : Dict
            DESCRIPTION.

        Returns
        -------
        list
            DESCRIPTION.

        """
        if self.mendeleev:
            atom_num_pairs = [[graph[Features.atom][pair[0]], graph[Features.atom][pair[1]]] for pair in graph[Features.pair_indices]]
            distance_features = Embedding_edges(converter=GaussianDistance(n=57)).embedding(graph[Features.bond])
            multi_properties = Embedding_edges(converter=MultiPropertyFeatures(self.properties)).embedding(atom_num_pairs)
            bond_features = np.concatenate([distance_features, multi_properties], axis=1)
            local_env = self._local_coordinates(graph)
        else:
            distance_features = Embedding_edges(converter=GaussianDistance()).embedding(graph[Features.bond])
            local_env = self._local_coordinates(graph)
            bond_features = distance_features
        
        return [
            np.array(graph[Features.atom], dtype=np.int32),
            np.array(bond_features),
            np.array(graph[Features.state], dtype=np.int32),
            np.array(graph[Features.pair_indices], dtype=np.int32),
            np.array(local_env),
            ]


    def get_graph(self, structure: Structure) -> Dict:
        """
        Parameters
        ----------
        structure : Structure
            DESCRIPTION.
        space_group_number : int
            DESCRIPTION.

        Raises
        ------
        RuntimeError
            DESCRIPTION.

        Returns
        -------
        dict
            DESCRIPTION.

        """
        lattice = np.array(structure.as_dict()['lattice']['matrix'], dtype=np.float32)
        cart_coords = structure.cart_coords.astype(np.float32)
        space_group_number = get_space_group_number(structure) - 1
        state_attributes = np.array([space_group_number], dtype="int32")
        center_indices, neighbor_indices, images, bonds = get_nn_info(structure, cutoff=self.cutoff)
        atoms = self.get_Z_number(structure)
        pair_indices = np.concatenate([center_indices, neighbor_indices], axis=0).reshape(2, -1).transpose()

        return {Features.atom: atoms, Features.bond: bonds, Features.state: state_attributes,
                Features.pair_indices: pair_indices, Features.image: images,
                Features.lattice: lattice, Features.cart_coords: cart_coords}

    
    def _local_coordinates(self, graph: Dict) -> np.ndarray: 
        """
        calculate local environment using numpy array only.
        Parameters
        ----------
        graph : Dict
            DESCRIPTION.

        Returns
        -------
        TYPE
            a seires of polar coordinates used for building local coordinate.

        """
        pair_indices = graph[Features.pair_indices]
        images = graph[Features.image]
        lattice = graph[Features.lattice]
        a, b, c = lattice[0], lattice[1], lattice[2]
        cart_coords = graph[Features.cart_coords]
        it1 = itemgetter(pair_indices[:,0])
        it2 = itemgetter(pair_indices[:,1])
        recive=it1(cart_coords)
        send=it2(cart_coords)
        polar = recive - send - np.expand_dims(images[:,0], axis=-1)*a - np.expand_dims(images[:,1], axis=-1)*b - np.expand_dims(images[:,2], axis=-1)*c
        zeros= np.ones_like(np.expand_dims(polar[:,1], axis=-1))
        vetical = np.concatenate([np.expand_dims(polar[:,1], axis=-1), -np.expand_dims(polar[:,0], axis=-1), zeros], axis=-1)
        local_env = np.concatenate([polar, vetical], axis=-1)
        return local_env


    def inputs_from_strcutre_list(self, structure_list: List) -> Set:
        """
        Parameters
        ----------
        structure_list : List
            a list of pymatgen.core.Structure
            In order to keep with Semantic space of atom2vector, MPRester should be used
            to get structures from materialsproject.

        Returns
        -------
        Set
            DESCRIPTION.

        """
        # Initialize graphs
        start = time.time()
        pool = Pool()
        graphs = pool.map(self.structure_to_input, structure_list)
        pool.close()
        pool.join()
        
        atom_features_list = []
        bond_features_list = []
        state_attrs_list = []
        pair_indices_list = []
        local_env_list = []
        for graph in graphs:
            atom_features, bond_features, state_attrs, pair_indices, local_env = graph
            atom_features_list.append(atom_features)
            bond_features_list.append(bond_features)
            state_attrs_list.append(state_attrs)
            pair_indices_list.append(pair_indices)
            local_env_list.append(local_env)
        
        end = time.time()
        run_time = end - start
        print('run time: ', run_time)
        
        return (
            atom_features_list,
            bond_features_list,
            local_env_list,
            state_attrs_list,
            pair_indices_list,
            )    
    
    
class  GraphBatchGeneratorSequence(Sequence):   
    def __init__(self, atom_features_list: List[np.ndarray],
                bond_features_list: List[np.ndarray],
                local_env_list: List[np.ndarray],
                state_attrs_list: List[np.ndarray],
                pair_indices_list: List[np.ndarray],
                labels: Union[List, None]=None,
                task_type=None,
                batch_size=32,
                is_shuffle=False):
        """
        Parameters
        ----------
        X_tensor : TYPE
            DESCRIPTION.
        y_label : TYPE
            DESCRIPTION.
        batch_size : TYPE, optional
            DESCRIPTION. The default is 64.
        is_shuffle : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        """
        self.task_type = task_type
        self.data_size = len(atom_features_list)
        self.batch_size = batch_size
        self.total_index = np.arange(self.data_size)

        self.atom_features_list = atom_features_list
        self.bond_features_list = bond_features_list
        self.local_env_list = local_env_list
        self.state_attrs_list = state_attrs_list
        self.pair_indices_list = pair_indices_list

        self.labels = labels

        if is_shuffle:
            shuffle = itemgetter(np.random.permutation(self.total_index))
            self.total_index = shuffle(self.total_index)
    
    
    def __len__(self) -> int:
        return int(np.ceil(self.data_size / self.batch_size))


    def on_epoch_end(self):
        """
        code to be executed on epoch end
        """
        self.total_index = np.random.permutation(self.total_index)


    def __getitem__(self, index: int) -> tuple:
        batch_index = self.total_index[index * self.batch_size : (index + 1) * self.batch_size]
        get = itemgetter(*batch_index)

        atom_features_list = get(self.atom_features_list)
        bond_features_list = get(self.bond_features_list)
        local_env_list =  get(self.local_env_list)
        state_attrs_list = get(self.state_attrs_list)
        pair_indices_list = get(self.pair_indices_list)

        inputs_batch = (atom_features_list,
                        bond_features_list,
                        local_env_list,
                        state_attrs_list,
                        pair_indices_list,
                        )

        x_batch = self._merge_batch(inputs_batch)
        if self.labels is None:
            return x_batch
        y_batch = np.array(get(self.labels))

        return x_batch, (y_batch)


    # def __getitem__(self, index: int) -> tuple:
    #     batch_index = self.total_index[index * self.batch_size : (index + 1) * self.batch_size]
    #     get = itemgetter(*batch_index)

    #     atom_features_list = list(get(self.atom_features_list))
    #     bond_features_list = list(get(self.bond_features_list))
    #     local_env_list =  list(get(self.local_env_list))
    #     state_attrs_list = list(get(self.state_attrs_list))
    #     pair_indices_list = list(get(self.pair_indices_list))

    #     inputs_batch = (tf.ragged.constant(atom_features_list, dtype=tf.int32),
    #         tf.ragged.constant(bond_features_list, dtype=tf.float32),
    #         tf.ragged.constant(local_env_list, dtype=tf.float32),
    #         tf.ragged.constant(state_attrs_list, dtype=tf.int32),
    #         tf.ragged.constant(pair_indices_list, dtype=tf.int64),
    #         )

    #     x_batch = self._merge_batch(inputs_batch)
    #     y_batch = np.atleast_2d(get(self.labels))

    #     return x_batch, y_batch


    def _merge_batch(self, x_batch: tuple) -> tuple:
        """
        Merging a batch of graphs into a disconnected graph should reindex atoms only
        features of graphs desn't be changed only merge them to one dimension of globl graph.
        reindex indices in pair_indices by adding increment of number of atoms in the batch
        atom marked with structure indice also need to be tell in globl graph.
        Parameters
        ----------
        x_batch : TYPE
            DESCRIPTION.

        Returns
        -------
        atom_features : TYPE
            DESCRIPTION.
        bond_features : TYPE
            DESCRIPTION.
        state_attributes : TYPE
            DESCRIPTION.
        pair_indices : TYPE
            DESCRIPTION.
        atom_partition_indices: TYPE
            DESCRIPTION.
        bond_partition_indices: TYPE
            DESCRIPTION.
        """
        atom_features, bond_features, local_env, state_attrs, pair_indices = x_batch
    
        # Obtain number of atoms and bonds for each graph
        # allocate graph (structure) indice for atom and bond in global graph
        num_atoms_per_graph = []
        atom_graph_indices = []
        for i, atoms in enumerate(atom_features):
            num = len(atoms)
            num_atoms_per_graph.append(num)
            atom_graph_indices += [i] * num

        atom_graph_indices = np.array(atom_graph_indices)

        num_bonds_per_graph = []
        bond_graph_indices = []
        for i, bonds in enumerate(bond_features):
            num = len(bonds)
            num_bonds_per_graph.append(num)
            bond_graph_indices += [i] * num

        bond_graph_indices = np.array(bond_graph_indices)
    
        # Increment is accumulative number of atom of each graph, it is used to reindex
        # indices of atom in global graph, so it should be adding to pair indices apart
        # from the first graph. The first subgraph keep its atom indices in global graph.
        # In order to add increment to pair indices, each accumulative number in increment
        # should be repeat num_bonds times so that every indice in pair_indices
        # accumulative number for first graph is zeros so that should pad num_bonds of zero to increment.
        increment = np.cumsum(num_atoms_per_graph[:-1])
        increment = np.pad(
            np.repeat(increment, num_bonds_per_graph[1:]), [(num_bonds_per_graph[0], 0)])
        
        pair_indices_per_graph = np.concatenate(pair_indices, axis=0)
        pair_indices = pair_indices_per_graph + np.expand_dims(increment, axis=-1)
        atom_features = np.concatenate(atom_features, axis=0)
        bond_features = np.concatenate(bond_features, axis=0)
        state_attrs = np.concatenate(state_attrs, axis=0)
        
        # Local spherical theta phi used for EdgeNetworks, the same as NodeNetworks.      
        local_env = np.concatenate(local_env, axis=0)
    
        return (atom_features, bond_features, local_env, state_attrs, pair_indices, atom_graph_indices,
                bond_graph_indices, pair_indices_per_graph)


class  GraphBatchGeneratorFromGraphs(GraphBatchGeneratorSequence):     
    def __init__(self, graphs: List, labels: List, task_type, batch_size=32):
        self.graphs = graphs
        self.labels = labels
        self.task_type = task_type
        self.batch_size = batch_size
        self.data_size = len(labels)


    def __getitem__(self, index: int) -> tuple:
        structure_batch = self.graphs[index * self.batch_size : (index + 1) * self.batch_size]
        y_batch = np.array(self.labels[index * self.batch_size : (index + 1) * self.batch_size])

        graph_batch = self._inputs_from_graphs(structure_batch)
        x_batch = self._merge_batch(graph_batch)
        return x_batch, (y_batch)


    def _inputs_from_graphs(self, graphs_list: List) -> Set:
        """
        Parameters
        ----------
        structure_list : List
            a list of pymatgen.core.Structure
            In order to keep with Semantic space of atom2vector, MPRester should be used
            to get structures from materialsproject.

        Returns
        -------
        Set
            DESCRIPTION.

        """
        # Initialize graphs
        atom_features_list = []
        bond_features_list = []
        state_attrs_list = []
        pair_indices_list = []
        local_env_list = []
        for s in graphs_list:
            atom_features, bond_features, state_attrs, pair_indices, local_env = s
            atom_features_list.append(atom_features)
            bond_features_list.append(bond_features)
            state_attrs_list.append(state_attrs)
            pair_indices_list.append(pair_indices)
            local_env_list.append(local_env)
  
        return (
            atom_features_list,
            bond_features_list,
            local_env_list,
            state_attrs_list,
            pair_indices_list,
            )


class  GraphBatchGeneratorBase:
    def __init__(self, X_tensor, y_label, task_type, batch_size=32, is_shuffle=False):
        """
        Parameters
        ----------
        X_tensor : TYPE
            DESCRIPTION.
        y_label : TYPE
            DESCRIPTION.
        batch_size : TYPE, optional
            DESCRIPTION. The default is 64.
        is_shuffle : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        """
        self.batch_size = batch_size
        self.task_type = task_type
        self.dataset = tf.data.Dataset.from_tensor_slices((X_tensor, (y_label)))
        if is_shuffle:
            self.dataset = self.dataset.shuffle(1024)
    
    
    def generate_dataset(self):
        """
        Returns
        -------
        TYPE
            partition datas into batches, and using merge_batch track graphs to a global graph.
        """
        return self.dataset.batch(self.batch_size).map(self._merge_batch, -1)


    def _merge_batch(self, x_batch, y_batch):
        """
        Merging a batch of graphs into a disconnected graph should reindex atoms only
        features of graphs desn't be changed only merge them to one dimension of globl graph.
        reindex indices in pair_indices by adding increment of number of atoms in the batch
        atom marked with structure indice also need to be tell in globl graph.
        Parameters
        ----------
        x_batch : TYPE
            DESCRIPTION.
        y_batch : TYPE
            DESCRIPTION.

        Returns
        -------
        atom_features : TYPE
            DESCRIPTION.
        bond_features : TYPE
            DESCRIPTION.
        state_attributes : TYPE
            DESCRIPTION.
        pair_indices : TYPE
            DESCRIPTION.
        atom_partition_indices: TYPE
            DESCRIPTION.
        bond_partition_indices: TYPE
            DESCRIPTION.
        y_batch : TYPE
            DESCRIPTION.

        """
        atom_features, bond_features, local_env, state_attrs, pair_indices = x_batch
    
        # Obtain number of atoms and bonds for each graph
        num_atoms_per_graph = atom_features.row_lengths()
        num_bonds_per_graph = bond_features.row_lengths()
        # max_num_atoms = tf.reduce_max(num_atoms_per_graph)
        
        # get adjacent matrix for each graph
        # adj_matrixes = self.adjacent_matrix_batch(max_num_atoms, pair_indices)
        
        # allocate graph (structure) indice for atom and bond in global graph
        graph_indices = tf.range(len(num_atoms_per_graph))
        atom_graph_indices = tf.repeat(graph_indices, num_atoms_per_graph)
        bond_graph_indices = tf.repeat(graph_indices, num_bonds_per_graph)
    
        # Increment is accumulative number of atom of each graph, it is used to reindex
        # indices of atom in global graph, so it should be adding to pair indices apart
        # from the first graph. The first subgraph keep its atom indices in global graph.
        # In order to add increment to pair indices, each accumulative number in increment
        # should be repeat num_bonds times so that every indice in pair_indices
        # accumulative number for first graph is zeros so that should pad num_bonds of zero to increment.
        increment = tf.cumsum(num_atoms_per_graph[:-1])
        increment = tf.pad(
            tf.repeat(increment, num_bonds_per_graph[1:]), [(num_bonds_per_graph[0], 0)])
        
        pair_indices_per_graph = pair_indices.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
        pair_indices = pair_indices_per_graph + tf.expand_dims(increment, axis=-1)
        atom_features = atom_features.merge_dims(outer_axis=0, inner_axis=1)
        bond_features = bond_features.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
        state_attrs = state_attrs.merge_dims(outer_axis=0, inner_axis=1)
        
        # Local spherical theta phi used for EdgeNetworks, the same as NodeNetworks.
        # num_edges_per_graph = local_env.row_lengths()
        # edge_graph_indices = tf.repeat(graph_indices, num_edges_per_graph)
        # increment_edges = tf.cumsum(num_bonds_per_graph[:-1])
        # increment_edges = tf.pad(
        #     tf.repeat(increment_edges, num_edges_per_graph[1:]), [(num_edges_per_graph[0], 0)])
        
        local_env = local_env.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
    
        return (atom_features, bond_features, local_env, state_attrs, pair_indices, atom_graph_indices,
                bond_graph_indices, pair_indices_per_graph), y_batch


class  GraphBatchGenerator(GraphBatchGeneratorBase):

    def _merge_batch(self, x_batch, y_batch):
        """
        Merging a batch of graphs into a disconnected graph should reindex atoms only
        features of graphs desn't be changed only merge them to one dimension of globl graph.
        reindex indices in pair_indices by adding increment of number of atoms in the batch
        atom marked with structure indice also need to be tell in globl graph.
        Parameters
        ----------
        x_batch : TYPE
            DESCRIPTION.
        y_batch : TYPE
            DESCRIPTION.

        Returns
        -------
        atom_features : TYPE
            DESCRIPTION.
        bond_features : TYPE
            DESCRIPTION.
        state_attributes : TYPE
            DESCRIPTION.
        pair_indices : TYPE
            DESCRIPTION.
        atom_partition_indices: TYPE
            DESCRIPTION.
        bond_partition_indices: TYPE
            DESCRIPTION.
        y_batch : TYPE
            DESCRIPTION.

        """
        atom_features, bond_features, local_env, state_attrs, pair_indices = x_batch
    
        # Obtain number of atoms and bonds for each graph
        num_atoms_per_graph = []
        for i in atom_features:
            num_atoms_per_graph.append(len(i))

        num_bonds_per_graph = []
        for i in bond_features:
            num_bonds_per_graph.append(len(i))
        # max_num_atoms = tf.reduce_max(num_atoms_per_graph)
        
        # allocate graph (structure) indice for atom and bond in global graph
        graph_indices = np.arange(len(num_atoms_per_graph))
        atom_graph_indices = np.repeat(graph_indices, num_atoms_per_graph)
        bond_graph_indices = np.repeat(graph_indices, num_bonds_per_graph)
    
        # Increment is accumulative number of atom of each graph, it is used to reindex
        # indices of atom in global graph, so it should be adding to pair indices apart
        # from the first graph. The first subgraph keep its atom indices in global graph.
        # In order to add increment to pair indices, each accumulative number in increment
        # should be repeat num_bonds times so that every indice in pair_indices
        # accumulative number for first graph is zeros so that should pad num_bonds of zero to increment.
        increment = np.cumsum(num_atoms_per_graph[:-1])
        increment = np.pad(
            np.repeat(increment, num_bonds_per_graph[1:]), [(num_bonds_per_graph[0], 0)])
        
        pair_indices_per_graph = np.concatenate(pair_indices, axis=0)
        pair_indices = pair_indices_per_graph + np.expand_dims(increment, axis=-1)
        atom_features = np.concatenate(atom_features, axis=0)
        bond_features = np.concatenate(bond_features, axis=0)
        state_attrs = np.concatenate(state_attrs, axis=0)

        # Local spherical theta phi used for EdgeNetworks, the same as NodeNetworks.        
        local_env = np.concatenate(local_env, axis=0)
    
        return (atom_features, bond_features, local_env, state_attrs, pair_indices, atom_graph_indices,
                bond_graph_indices, pair_indices_per_graph), y_batch