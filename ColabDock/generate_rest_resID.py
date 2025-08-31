
import os, io, sys
import dataclasses
import argparse
import random
import joblib
import numpy as np
from Bio.PDB.PDBParser import PDBParser

import residue_constants
from config_7JTO_ColabDock import config

@dataclasses.dataclass(frozen=True)
class Protein:
  """Protein structure representation."""

  # Cartesian coordinates of atoms in angstroms. The atom types correspond to
  # residue_constants.atom_types, i.e. the first three are N, CA, CB.
  atom_positions: np.ndarray  # [num_res, num_atom_type, 3]

  # Amino-acid type for each residue represented as an integer between 0 and
  # 20, where 20 is 'X'.
  aatype: np.ndarray  # [num_res]

  # Binary float mask to indicate presence of a particular atom. 1.0 if an atom
  # is present and 0.0 if not. This should be used for loss masking.
  atom_mask: np.ndarray  # [num_res, num_atom_type]

  # Residue index as used in PDB. It is not necessarily continuous or 0-indexed.
  residue_index: np.ndarray  # [num_res]

  # B-factors, or temperature factors, of each residue (in sq. angstroms units),
  # representing the displacement of the residue from its ground truth mean
  # value.
  b_factors: np.ndarray  # [num_res, num_atom_type]


MODRES = {'MSE':'MET','MLY':'LYS','FME':'MET','HYP':'PRO',
          'TPO':'THR','CSO':'CYS','SEP':'SER','M3L':'LYS',
          'HSK':'HIS','SAC':'SER','PCA':'GLU','DAL':'ALA',
          'CME':'CYS','CSD':'CYS','OCS':'CYS','DPR':'PRO',
          'B3K':'LYS','ALY':'LYS','YCM':'CYS','MLZ':'LYS',
          '4BF':'TYR','KCX':'LYS','B3E':'GLU','B3D':'ASP',
          'HZP':'PRO','CSX':'CYS','BAL':'ALA','HIC':'HIS',
          'DBZ':'ALA','DCY':'CYS','DVA':'VAL','NLE':'LEU',
          'SMC':'CYS','AGM':'ARG','B3A':'ALA','DAS':'ASP',
          'DLY':'LYS','DSN':'SER','DTH':'THR','GL3':'GLY',
          'HY3':'PRO','LLP':'LYS','MGN':'GLN','MHS':'HIS',
          'TRQ':'TRP','B3Y':'TYR','PHI':'PHE','PTR':'TYR',
          'TYS':'TYR','IAS':'ASP','GPL':'LYS','KYN':'TRP',
          'CSD':'CYS','SEC':'CYS'}

def pdb_to_string(pdb_file):
    modres = {**MODRES}
    lines = []
    for line in open(pdb_file,"rb"):
        line = line.decode("utf-8","ignore").rstrip()
        if line[:6] == "MODRES":
            k = line[12:15]
            v = line[24:27]
            if k not in modres and v in residue_constants.restype_3to1:
                modres[k] = v
        if line[:6] == "HETATM":
            k = line[17:20]
            if k in modres:
                line = "ATOM  "+line[6:17]+modres[k]+line[20:]
        if line[:4] == "ATOM":
            lines.append(line)
    return "\n".join(lines)

def from_pdb_string(pdb_str: str, chain_id) -> Protein:
    """Takes a PDB string and constructs a Protein object.

    WARNING: All non-standard residue types will be converted into UNK. All
    non-standard atoms will be ignored.

    Args:
      pdb_str: The contents of the pdb file
      chain_id: If None, then the pdb file must contain a single chain (which
      will be parsed). If chain_id is specified (e.g. A), then only that chain
      is parsed.

    Returns:
    A new `Protein` parsed from the pdb contents.
    """
    pdb_fh = io.StringIO(pdb_str)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('none', pdb_fh)
    models = list(structure.get_models())
    if len(models) != 1:
        raise ValueError(
            f'Only single model PDBs are supported. Found {len(models)} models.')
    model = models[0]

    if chain_id is not None:
        chain = model[chain_id]
    else:
        chains = list(model.get_chains())
        if len(chains) != 1:
            raise ValueError(
                'Only single chain PDBs are supported when chain_id not specified. '
                f'Found {len(chains)} chains.')
        else:
            chain = chains[0]

    atom_positions = []
    aatype = []
    atom_mask = []
    residue_index = []
    b_factors = []

    for res in chain:
        if res.id[2] != ' ':
            raise ValueError(
                f'PDB contains an insertion code at chain {chain.id} and residue '
                f'index {res.id[1]}. These are not supported.')
        res_shortname = residue_constants.restype_3to1.get(res.resname, 'X')
        restype_idx = residue_constants.restype_order.get(
                        res_shortname, residue_constants.restype_num)
        pos = np.zeros((residue_constants.atom_type_num, 3))
        mask = np.zeros((residue_constants.atom_type_num,))
        res_b_factors = np.zeros((residue_constants.atom_type_num,))
        for atom in res:
            if atom.name not in residue_constants.atom_types:
                continue
            pos[residue_constants.atom_order[atom.name]] = atom.coord
        mask[residue_constants.atom_order[atom.name]] = 1.
        res_b_factors[residue_constants.atom_order[atom.name]] = atom.bfactor
        if np.sum(mask) < 0.5:
            # If no known atom positions are reported for the residue then skip it.
            continue
        aatype.append(restype_idx)
        atom_positions.append(pos)
        atom_mask.append(mask)
        residue_index.append(res.id[1])
        b_factors.append(res_b_factors)

    return Protein(atom_positions=np.array(atom_positions),
                   atom_mask=np.array(atom_mask),
                   aatype=np.array(aatype),
                   residue_index=np.array(residue_index),
                   b_factors=np.array(b_factors))

def prep_pdb(pdb_filename, chain=None):
    '''extract features from pdb'''

    #print('chain')
    #print(chain)
    # go through each defined chain
    chains = [None] if chain is None else chain.split(",")
    #print('chains')
    #print(chains)
    o = {}
    o['residue_index'] = []
    last = 0
    residue_idx, chain_idx = [],[]
    for chain in chains:
        #print('chain:')
        #print(chain)
        protein_obj = from_pdb_string(pdb_to_string(pdb_filename), chain_id=chain)
        batch = {'aatype': protein_obj.aatype,
                 'all_atom_positions': protein_obj.atom_positions,
                 'all_atom_mask': protein_obj.atom_mask}

        #print(batch)
        has_ca = batch["all_atom_mask"][:,0] == 1
        #print(has_ca)

        #print('residue_idx:')
        #print(residue_idx)
        #print('last:')
        #print(last)
        #print('protein_obj.residue_index:')
        #print(protein_obj.residue_index)
        residue_index = protein_obj.residue_index + last
        last = residue_index[-1] + 50
        #print('last:')
        #print(last)
        #print('residue_idx:')
        #print(residue_idx)

        o['residue_index'].append(residue_index)
        residue_idx.append(protein_obj.residue_index)
        chain_idx.append([chain] * len(residue_idx[-1]))

    o['idx'] = {"residue":np.concatenate(residue_idx), "chain":np.concatenate(chain_idx)}
    #print('o:')
    #print(o)

    return o

if __name__ == '__main__':
    description = 'convert the res ID in PDB file to nciwubvuyebhubdhbcuqwbicvnmn'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('pdb', type=str, help='input complex structure')
    parser.add_argument('rest_type', type=str, help='sampled restraints type. Please choose one type from 1vN and MvN')
    parser.add_argument('--save_path', type=str, default=None, help='the file to save the sampled restraints using joblib package. Default None.')

    # parse argument
    args = parser.parse_args()
    file = args.pdb
    chains = config.chains
    chains_rest = config.chains
    resID_pocket_receptor = config.resID_pocket_receptor
    resID_pocket_ligand = config.resID_pocket_ligand
    rest_type = args.rest_type
    save_path = args.save_path

    N_MvN = 10

    print('resID_pocket_receptor:')
    print(len(resID_pocket_receptor))
    print(resID_pocket_receptor)
    print('resID_pocket_ligand:')
    print(len(resID_pocket_ligand))
    print(resID_pocket_ligand)

    # check argument
    if not os.path.exists(file):
        raise Exception('The pdb file you provide does not exist!')

    if rest_type not in ['1vN', 'MvN']:
        raise Exception('The rest_type argument accepts 1vN or MvN.')

    chains_all_l = [c.strip() for c in chains.split(',')]
    chains_rest_l = [c.strip() for c in chains_rest.split(',')]

    pdb_parser = PDBParser(QUIET=True)
    structures = pdb_parser.get_structure('none', file)
    structure = list(structures.get_models())[0]
    for ichain in chains_all_l:
        if ichain not in structure:
            raise Exception(f'Chain {ichain} is not in the provided pdb file!')

    for ichain in chains_rest_l:
        if ichain not in chains_all_l:
            raise Exception(f'Chain {ichain} in chains_rest is not in chains!')

    if len(set(chains_rest_l)) != 2 or len(chains_rest_l) != 2:
        raise Exception('Currently, this script only generates restraints between two chains.')

    # cal distance matrix
    pdb = prep_pdb(file, chain=chains)
    #print('pdb:')
    #print(pdb)

    # sample restraints
    lens = [(pdb["idx"]["chain"] == c).sum() for c in chains_all_l]
    boundaries = [0] + list(np.cumsum(lens))
    ind = chains_all_l.index(chains_rest_l[0])
    a_start, a_stop = boundaries[ind], boundaries[ind+1]
    print(lens)
    print(boundaries)
    print(ind)
    print(a_start)
    print(a_stop)
    print('**********')
    ind = chains_all_l.index(chains_rest_l[1])
    b_start, b_stop = boundaries[ind], boundaries[ind+1]
    print(ind)
    print(a_start)
    print(a_stop)

    pdb_residue_index = np.concatenate(pdb['residue_index'])
    print('pdb_residue_index:')
    print(pdb_residue_index)
    resID_inPDB_pocket_receptor = []
    resID_inPDB_pocket_ligand = []
    print('resID_pocket_receptor:')
    for resID in resID_pocket_receptor:
        positions_in_pdb = np.where(pdb['idx']['residue'] == resID)
        #print(positions_in_pdb)
        positions = np.where(pdb['idx']['chain'][positions_in_pdb] == chains_all_l[0])
        resID_inPDB_pocket_receptor.append(positions_in_pdb[0][positions[0][0]])
        #print(pdb_residue_index[positions_in_pdb[0][positions[0][0]]])
    print('resID_pocket_ligand:')
    for resID in resID_pocket_ligand:
        positions_in_pdb = np.where(pdb['idx']['residue'] == resID)
        #print(positions_in_pdb)
        positions = np.where(pdb['idx']['chain'][positions_in_pdb] == chains_all_l[1])
        resID_inPDB_pocket_ligand.append(positions_in_pdb[0][positions[0][0]])
        #print(pdb_residue_index[positions_in_pdb[0][positions[0][0]]])
    print('resID_inPDB_pocket_receptor:')
    print(resID_inPDB_pocket_receptor)
    print('resID_inPDB_pocket_ligand:')
    print(resID_inPDB_pocket_ligand)
    #sys.exit()

    if rest_type == '1vN':
        num = 10
    elif rest_type == 'MvN':
        rest1 = []
        for i in range(len(resID_inPDB_pocket_receptor)):
            iMvN = []
            iMvN.append([resID_inPDB_pocket_receptor[i], resID_inPDB_pocket_ligand])
            rest1.append(iMvN)
        rest1.append(N_MvN)
        rest2 = []
        for i in range(len(resID_inPDB_pocket_ligand)):
            iMvN = []
            iMvN.append([resID_inPDB_pocket_ligand[i], resID_inPDB_pocket_receptor])
            rest2.append(iMvN)
        rest2.append(N_MvN)

    with open('restraint.txt', 'w') as f:
        f.write("    'rest_MvN': [[")
        for i, re in enumerate(rest1):
            if i != 0:
                f.write('                 ')
            if re == N_MvN:
                f.write(str(re))
                f.write('],\n')
                continue
            for r in re:
                f.write(str(r))
                f.write(',\n')
        for i, re in enumerate(rest2):
            f.write('                 ')
            if i == 0:
                f.write('[')
            if re == N_MvN:
                f.write(str(re))
                f.write(']],\n')
                continue
            for r in re:
                f.write(str(r))
                f.write(',\n')

    if save_path is not None:
        joblib.dump(rest1, save_path)
    else:
        print(f'The {rest_type} restraints:\n{rest1}')


