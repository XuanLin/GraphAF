# coding=utf-8
"""
Anonymous author
# Part of the codes are taken from source code of chainer-chemistry.
Description: load raw smiles, construct node/edge matrix.
"""

import sys
import os
import argparse

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops


def check_num_atoms(mol, num_max_atoms=-1):
    """
    Check number of atoms in mol does not exceed num_max_atoms
    If number of atoms in mol exceeds the number num_max_atoms, it will return False
    Args:
        mol (rdkit.Chem.Mol):
        num_max_atoms (int): If negative value is set, do not check number, return True.
    Returns:
        bool value
    """
    num_atoms = mol.GetNumAtoms()
    if num_max_atoms >= 0 and num_atoms > num_max_atoms:
        return False
    return True


def construct_atomic_number_array(mol, max_size=-1):
    """
    Returns atomic numbers of atoms in a molecule.
    Args:
        mol (rdkit.Chem.Mol): Input molecule.
        max_size (int): The size of returned array.
            If max_size is negative, return the atomic array of original molecule.
            Otherwise, it must be larger than the number of atoms
            in the input molecules. In that case, the tail of
            the array is padded with zeros.
    Returns:
        np.ndarray: an array consisting of atomic numbers
            of atoms in the molecule.
    """

    atom_list = [a.GetAtomicNum() for a in mol.GetAtoms()]
    n_atom = len(atom_list)

    if max_size < 0:
        return np.array(atom_list, dtype=np.int32)
    elif max_size >= n_atom:
        # zero padding for atom_list
        # 0 represents padding atom
        atom_array = np.zeros(max_size, dtype=np.int32)
        atom_array[:n_atom] = np.array(atom_list, dtype=np.int32)
        return atom_array
    else:
        raise ValueError('max_size (%d) must be negative '
                         'or no less than the number of atoms '
                         'in the input molecule (%d)' % (max_size, n_atom))


def construct_discrete_edge_matrix(mol, max_size=-1):
    """
    Returns the edge-type dependent adjacency matrix of the given molecule.
    Args:
        mol (rdkit.Chem.Mol): Input molecule.
        max_size (int): The size of the returned matrix.
            If max_size is negative, return the edge matrix of original molecule.
            Otherwise, it must be larger than the number of atoms
            in the input molecules. In that case, the adjacent
            matrix is expanded and zeros are padded to right
            columns and bottom rows.

    Returns:
        adj_array (np.ndarray): The adjacent matrix of the input molecule.
            It is 3-dimensional array with shape (edge_type, atoms1, atoms2),
            where edge_type represents the bond type,
            atoms1 & atoms2 represent from and to of the edge respectively.
            If max_size is non-negative, its size is equal to that value.
            Otherwise, it is equal to the number of atoms in the molecule.
    """
    if mol is None:
        raise ValueError('mol is None')
    N = mol.GetNumAtoms()

    if max_size < 0:
        size = N
    elif max_size >= N:
        size = max_size
    else:
        raise ValueError(
            'max_size (%d) is smaller than number of atoms in mol (%d)' % (max_size, N))
    adjs = np.zeros((4, size, size), dtype=np.float32)

    # Acutually, we first kekulize each molecule, so there is no aromatic bond in molecule.
    # For robustness, we keep the aromatic bond type.
    bond_type_to_channel = {
        Chem.BondType.SINGLE: 0,
        Chem.BondType.DOUBLE: 1,
        Chem.BondType.TRIPLE: 2,
        Chem.BondType.AROMATIC: 3
    }
    for bond in mol.GetBonds():
        bond_type = bond.GetBondType()
        ch = bond_type_to_channel[bond_type]
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        adjs[ch, i, j] = 1.0
        adjs[ch, j, i] = 1.0
    return adjs


class SmilesPreprocessor(object):
    """
    preprocessor class specified for rdkit mol instance

    Initialize args:
        add_Hs: add hydrogen onto mol
        kekulize: kekulize molecule. Convert aromatic bond to single/double bond
        max_atoms: ignore molecule whose atoms is more than max_atoms in dataset
        max_size: output size of vector/matrix, used for padding.
    """
    def __init__(self, add_Hs=False, kekulize=True, max_atoms=-1, max_size=-1):
        self.add_Hs = add_Hs
        self.kekulize = kekulize
        self.max_atoms = max_atoms
        self.max_size = max_size
        assert (max_atoms < 0 or max_size < 0 or max_atoms <= max_size)

    def _prepare_mol(self, Smiles):
        """
        Get mol from Smiles, add Hs and kekulize
        """
        mol = Chem.MolFromSmiles(Smiles)
        canonical_smiles = Chem.MolToSmiles(mol)
        if self.add_Hs:
            mol = Chem.AddHs(mol)
        if self.kekulize:
            Chem.Kekulize(mol)
        return mol, canonical_smiles

    def _get_features(self, mol):
        """
        Get atomic number array and discrete edge matrix
        """
        if not check_num_atoms(mol, self.max_atoms):
            return None, None, None
        atom_array = construct_atomic_number_array(mol, self.max_size)
        adj_array = construct_discrete_edge_matrix(mol, self.max_size)
        mol_size = mol.GetNumAtoms()
        return atom_array, adj_array, mol_size

    def process(self, smiles):
        mol, canonical_smiles = self._prepare_mol(smiles)
        atom_array, adj_array, mol_size = self._get_features(mol)
        return atom_array, adj_array, mol_size, canonical_smiles

#TODO: add Qm9 processor
class Qm9_Processor(object):
    pass


class Zinc_Processor(object):
    def __init__(self, in_path, out_path, freedom=0):
        self.in_path = in_path
        self.out_path = out_path
        self.freedom = freedom

        # C N O F P S Cl Br I virtual
        self.atom_list = [6, 7, 8, 9, 15, 16, 17, 35, 53, 0]
        self.node_dim = len(self.atom_list)
        self.max_size = 38 + self.freedom # we allow generating molecule with no more than 45 atoms.
        self.n_bond = 3 # single/double/triple
        self.smiles_processor = SmilesPreprocessor(add_Hs=False, kekulize=True, max_atoms=38, max_size=self.max_size)
        self.node_features, self.adj_features, self.mol_sizes = self._load_data(self.in_path)
        self._save_data(self.out_path)
        self._save_config(self.out_path)
    

    def _load_data(self, path):
        """
        Read smiles from data stored in path. preprocess using smiles_processor
        """
        cnt = 0
        all_node_feature = []
        all_adj_feature = []
        all_mol_size = []
        fp = open(path, 'r')
        for smiles in fp:
            smiles = smiles.strip()
            atom_array, adj_array, mol_size, canonical_smiles = self.smiles_processor.process(smiles)
            if atom_array is not None:
                cnt += 1
                if cnt % 10000 == 0:
                    print('current cnt: %d' % cnt)
               

                all_node_feature.append(atom_array)
                all_adj_feature.append(adj_array[:3])
                all_mol_size.append(mol_size)
        fp.close()                
        self.n_molecule = cnt
        print('total number of valid molecule in dataset: %d' % self.n_molecule)
        return (np.array(all_node_feature), np.array(all_adj_feature), np.array(all_mol_size))


    def _save_data(self, path):
        print('saving node/adj feature...')
        print('shape of node feature:', self.node_features.shape)
        print('shape of adj features:', self.adj_features.shape)
        print('shape of mol sizes:', self.mol_sizes.shape)

        np.save(path + '_node_features', self.node_features)
        np.save(path + '_adj_features', self.adj_features.astype(np.uint8)) # save space
        np.save(path + '_mol_sizes', self.mol_sizes) # save space


    def _save_config(self, path):
        fp = open(path + '_config.txt', 'w')
        config = dict()
        config['atom_list'] = self.atom_list
        config['freedom'] = self.freedom
        config['node_dim'] = self.node_dim
        config['max_size'] = self.max_size
        config['bond_dim'] = self.n_bond + 1
        print('saving config...')
        print(config)
        fp.write(str(config))
        fp.close()


class Zinc_Processor_800(object):
    def __init__(self, in_path, out_path, freedom=0):
        self.in_path = in_path
        self.out_path = out_path
        self.freedom = freedom

        # C N O F P S Cl Br I virtual
        self.atom_list = [6, 7, 8, 9, 15, 16, 17, 35, 53, 0]
        self.node_dim = len(self.atom_list)
        self.max_size = 38 + self.freedom # we allow generating molecule with no more than 45 atoms.
        self.n_bond = 3 # single/double/triple
        self.smiles_processor = SmilesPreprocessor(add_Hs=False, kekulize=True, max_atoms=38, max_size=self.max_size)
        self.node_features, self.adj_features, self.mol_sizes = self._load_data(self.in_path)
        self._save_data(self.out_path)
        self._save_config(self.out_path)
    

    def _load_data(self, path):
        """
        Read smiles from data stored in path. preprocess using smiles_processor
        """
        cnt = 0
        all_node_feature = []
        all_adj_feature = []
        all_mol_size = []
        fp = open(path, 'r')
        for smiles in fp:
            smiles = smiles.strip()
            plogp, smiles = smiles.split(',')
            plogp = float(plogp)

            atom_array, adj_array, mol_size, canonical_smiles = self.smiles_processor.process(smiles)
            if atom_array is not None:
                cnt += 1
                if cnt % 10000 == 0:
                    print('current cnt: %d' % cnt)
               

                all_node_feature.append(atom_array)
                all_adj_feature.append(adj_array[:3])
                all_mol_size.append(mol_size)
        fp.close()                
        self.n_molecule = cnt
        print('total number of valid molecule in dataset: %d' % self.n_molecule)
        return (np.array(all_node_feature), np.array(all_adj_feature), np.array(all_mol_size))


    def _save_data(self, path):
        print('saving node/adj feature...')
        print('shape of node feature:', self.node_features.shape)
        print('shape of adj features:', self.adj_features.shape)
        print('shape of mol sizes:', self.mol_sizes.shape)

        np.save(path + '_node_features', self.node_features)
        np.save(path + '_adj_features', self.adj_features.astype(np.uint8)) # save space
        np.save(path + '_mol_sizes', self.mol_sizes) # save space


    def _save_config(self, path):
        fp = open(path + '_config.txt', 'w')
        config = dict()
        config['atom_list'] = self.atom_list
        config['freedom'] = self.freedom
        config['node_dim'] = self.node_dim
        config['max_size'] = self.max_size
        config['bond_dim'] = self.n_bond + 1
        print('saving config...')
        print(config)
        fp.write(str(config))
        fp.close()


class Zinc_Processor_Positive(object):
    def __init__(self, in_path, out_path, freedom=0):
        self.in_path = in_path
        self.out_path = out_path
        self.freedom = freedom

        # C N O F P S Cl Br I virtual
        self.atom_list = [6, 7, 8, 9, 15, 16, 17, 35, 53, 0]
        self.node_dim = len(self.atom_list)
        self.max_size = 38 + self.freedom # we allow generating molecule with no more than 45 atoms.
        self.n_bond = 3 # single/double/triple
        self.smiles_processor = SmilesPreprocessor(add_Hs=False, kekulize=True, max_atoms=38, max_size=self.max_size)
        self.node_features, self.adj_features, self.mol_sizes = self._load_data(self.in_path)
        self._save_data(self.out_path)
        self._save_config(self.out_path)
    

    def _load_data(self, path):
        """
        Read smiles from data stored in path. preprocess using smiles_processor
        """
        cnt = 0
        all_node_feature = []
        all_adj_feature = []
        all_mol_size = []
        fp = open(path, 'r')
        for smiles in fp:
            smiles = smiles.strip()

            atom_array, adj_array, mol_size, canonical_smiles = self.smiles_processor.process(smiles)
            if atom_array is not None:
                cnt += 1
                if cnt % 10000 == 0:
                    print('current cnt: %d' % cnt)
               

                all_node_feature.append(atom_array)
                all_adj_feature.append(adj_array[:3])
                all_mol_size.append(mol_size)
        fp.close()                
        self.n_molecule = cnt
        print('total number of valid molecule in dataset: %d' % self.n_molecule)
        return (np.array(all_node_feature), np.array(all_adj_feature), np.array(all_mol_size))


    def _save_data(self, path):
        print('saving node/adj feature...')
        print('shape of node feature:', self.node_features.shape)
        print('shape of adj features:', self.adj_features.shape)
        print('shape of mol sizes:', self.mol_sizes.shape)

        np.save(path + '_node_features', self.node_features)
        np.save(path + '_adj_features', self.adj_features.astype(np.uint8)) # save space
        np.save(path + '_mol_sizes', self.mol_sizes) # save space


    def _save_config(self, path):
        fp = open(path + '_config.txt', 'w')
        config = dict()
        config['atom_list'] = self.atom_list
        config['freedom'] = self.freedom
        config['node_dim'] = self.node_dim
        config['max_size'] = self.max_size
        config['bond_dim'] = self.n_bond + 1
        print('saving config...')
        print(config)
        fp.write(str(config))
        fp.close()



class Zinc_Processor_Negative(object):
    def __init__(self, in_path, out_path, freedom=0):
        self.in_path = in_path
        self.out_path = out_path
        self.freedom = freedom

        # C N O F P S Cl Br I virtual
        self.atom_list = [6, 7, 8, 9, 15, 16, 17, 35, 53, 0]
        self.node_dim = len(self.atom_list)
        self.max_size = 38 + self.freedom # we allow generating molecule with no more than 45 atoms.
        self.n_bond = 3 # single/double/triple
        self.smiles_processor = SmilesPreprocessor(add_Hs=False, kekulize=True, max_atoms=38, max_size=self.max_size)
        self.node_features, self.adj_features, self.mol_sizes = self._load_data(self.in_path)
        self._save_data(self.out_path)
        self._save_config(self.out_path)
    

    def _load_data(self, path):
        """
        Read smiles from data stored in path. preprocess using smiles_processor
        """
        cnt = 0
        all_node_feature = []
        all_adj_feature = []
        all_mol_size = []
        fp = open(path, 'r')
        for smiles in fp:
            smiles = smiles.strip()

            atom_array, adj_array, mol_size, canonical_smiles = self.smiles_processor.process(smiles)
            if atom_array is not None:
                cnt += 1
                if cnt % 10000 == 0:
                    print('current cnt: %d' % cnt)
               

                all_node_feature.append(atom_array)
                all_adj_feature.append(adj_array[:3])
                all_mol_size.append(mol_size)
        fp.close()                
        self.n_molecule = cnt
        print('total number of valid molecule in dataset: %d' % self.n_molecule)
        return (np.array(all_node_feature), np.array(all_adj_feature), np.array(all_mol_size))


    def _save_data(self, path):
        print('saving node/adj feature...')
        print('shape of node feature:', self.node_features.shape)
        print('shape of adj features:', self.adj_features.shape)
        print('shape of mol sizes:', self.mol_sizes.shape)

        np.save(path + '_node_features', self.node_features)
        np.save(path + '_adj_features', self.adj_features.astype(np.uint8)) # save space
        np.save(path + '_mol_sizes', self.mol_sizes) # save space


    def _save_config(self, path):
        fp = open(path + '_config.txt', 'w')
        config = dict()
        config['atom_list'] = self.atom_list
        config['freedom'] = self.freedom
        config['node_dim'] = self.node_dim
        config['max_size'] = self.max_size
        config['bond_dim'] = self.n_bond + 1
        print('saving config...')
        print(config)
        fp.write(str(config))
        fp.close()


if __name__ == '__main__':
    if len(sys.argv) != 5:
        raise ValueError('Please specify the dataset name, in_path, out_path and freedom in turn')
    data_name = sys.argv[1]
    in_path = sys.argv[2]
    out_path = sys.argv[3]
    freedom = int(sys.argv[4])

    processor_dict = {'zinc250k': Zinc_Processor,
                      'zinc250k_800': Zinc_Processor_800,
                      'zinc250k_pos1w': Zinc_Processor_Positive,
                      'zinc250k_neg1w': Zinc_Processor_Negative,
                     }
    
    processor = processor_dict[data_name](in_path, out_path, freedom=freedom)


