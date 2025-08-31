
# https://userguide.mdanalysis.org/stable/examples/analysis/trajectory_similarity/clustering_ensemble_similarity.html
import numpy as np
import matplotlib.pyplot as plt

import MDAnalysis as mda
from MDAnalysis.analysis import encore
from MDAnalysis.analysis.encore.clustering import ClusteringMethod as clm
from MDAnalysis.analysis import align
from MDAnalysis.analysis import distances

from scipy.spatial.transform import Rotation

from sklearn.linear_model import LinearRegression

from rdkit import Chem
from rdkit.Chem import AllChem
#from rdkit.Chem import rdDetermineBonds

import os, sys
import glob

import multiprocessing
import time


def add_linker_for_PROTAC(blockslice, clustered_pdb_list, index_list, return_dict):

    PROTAC_folder = 'PROTACs_for_DSDP'

    #for filename in clustered_pdb_list[blockslice.start:blockslice.stop]:
    #    ob_cmd = 'obabel ' + './' + filename + 'qt' + ' -O ' + './' + filename
    #    os.system(ob_cmd)

    aligned_traj = mda.Universe(clustered_pdb_list[0], clustered_pdb_list[blockslice.start:blockslice.stop])

    u_POI_ref = mda.Universe('./8qw7_A.pdb', './8qw7_A.pdb')
    u_E3_ref = mda.Universe('./8qw7_B.pdb', './8qw7_B.pdb')


    #PROTAC_smiles = 'N#Cc1c2c(CCCC2(c3onc(c4nc(N5CCCN(CC5C)CCCCOc6cc(C(C(N7C(CC(C7)O)C(NCc8ccc(c9scnc9C)cc8)=O)=O)C(C)C)on6)ncc4)n3)C)sc1N'
    PROTAC_smiles = 'N#Cc1c(N)sc2c1[C@](c3nc(c4ccnc(N5CCCN(CCCCOc6noc([C@@H](C(C)C)C(N7[C@H](C(NCc8ccc(c9c(C)ncs9)cc8)=O)C[C@@H](O)C7)=O)c6)C[C@@H]5C)n4)no3)(C)CCC2'
    POI_head_file = 'PROTAC_POI_head.pdb'
    E3_head_file = 'PROTAC_E3_head.pdb'

    u_PROTAC_POI_head = mda.Universe('./PROTAC_POI_head.pdb', './PROTAC_POI_head.pdb')
    u_PROTAC_E3_head = mda.Universe('./PROTAC_E3_head.pdb', './PROTAC_E3_head.pdb')
    PROTAC_POI_head = u_PROTAC_POI_head.select_atoms('all')
    PROTAC_E3_head = u_PROTAC_E3_head.select_atoms('all')

    print(len(aligned_traj.trajectory))


    sel_E3_CA = '(name CA)' + ' and (chainID B)'
    sel_POI_CA = '(name CA)' + ' and (chainID A)'
    sel_E3_all = 'all' + ' and (chainID B)'
    sel_POI_all = 'all' + ' and (chainID A)'

    E3_CA_ref = u_E3_ref.select_atoms(sel_E3_CA)
    POI_CA_ref = u_POI_ref.select_atoms(sel_POI_CA)
    E3_all_ref = u_E3_ref.select_atoms(sel_E3_all)
    POI_all_ref = u_POI_ref.select_atoms(sel_POI_all)

    E3_CA = aligned_traj.select_atoms(sel_E3_CA)
    POI_CA = aligned_traj.select_atoms(sel_POI_CA)
    E3_all = aligned_traj.select_atoms(sel_E3_all)
    POI_all = aligned_traj.select_atoms(sel_POI_all)


    timeseries = []
    E3_CA_ref_positions = E3_CA_ref.positions
    POI_CA_ref_positions = POI_CA_ref.positions
    E3_ref_COM = E3_all_ref.center_of_mass()
    POI_ref_COM = POI_all_ref.center_of_mass()

    E3_CA_ref_centered = E3_CA_ref_positions - E3_ref_COM
    POI_CA_ref_centered = POI_CA_ref_positions - POI_ref_COM

    PROTAC_E3_head_positions_backup = PROTAC_E3_head.positions
    PROTAC_POI_head_positions_backup = PROTAC_POI_head.positions

    box_info_list = []

    for ts in aligned_traj.trajectory:

        dist_arr = distances.distance_array(POI_all.positions,
                                            E3_all.positions)
        count_close_contact = np.count_nonzero(dist_arr < 2)
        #print('Time: ' + str(ts.frame))
        #print('Count of close contact: ' + str(count_close_contact))
        #if ts.frame > 100:
        #    break
        if count_close_contact > 0:
            continue

        #continue

        E3_COM = np.sum(E3_all.positions, axis=0) / len(E3_all.positions)
        POI_COM = np.sum(POI_all.positions, axis=0) / len(POI_all.positions)

        E3_CA_centered = E3_CA.positions - E3_COM
        POI_CA_centered = POI_CA.positions - POI_COM

        R_E3 = align.rotation_matrix(E3_CA_ref_centered,
                                     E3_CA_centered)[0]
        print('R_E3:')
        print(R_E3)
        R_POI = align.rotation_matrix(POI_CA_ref_centered,
                                      POI_CA_centered)[0]
        print('R_POI:')
        print(R_POI)

        R_obj_E3 = Rotation.from_matrix(R_E3)
        R_obj_POI = Rotation.from_matrix(R_POI)

        PROTAC_E3_head.atoms.translate(-E3_ref_COM)
        PROTAC_POI_head.atoms.translate(-POI_ref_COM)
        #PROTAC_E3_head.atoms.write("PROTAC_E3_head_translated.pdb")
        #PROTAC_POI_head.atoms.write("PROTAC_POI_head_translated.pdb")

        PROTAC_E3_head.atoms.rotate(R_E3)
        PROTAC_POI_head.atoms.rotate(R_POI)
        #PROTAC_E3_head.atoms.write("PROTAC_E3_head_rotated.pdb")
        #PROTAC_POI_head.atoms.write("PROTAC_POI_head_rotated.pdb")
        PROTAC_E3_head.atoms.translate(E3_COM)
        PROTAC_POI_head.atoms.translate(POI_COM)

        box_min_E3 = np.min(PROTAC_E3_head.positions, axis=0)
        box_min_POI = np.min(PROTAC_POI_head.positions, axis=0)
        box_max_E3 = np.max(PROTAC_E3_head.positions, axis=0)
        box_max_POI = np.max(PROTAC_POI_head.positions, axis=0)

        box_info = []
        box_info.append([index_list[blockslice.start + ts.frame]])
        box_info.append(np.min([box_min_E3, box_min_POI], axis=0) - 10)
        box_info.append(np.max([box_max_E3, box_max_POI], axis=0) + 10)

        #PROTAC_E3_head.atoms.write('PROTAC_E3_head_final.pdb')
        #PROTAC_POI_head.atoms.write('PROTAC_POI_head_final.pdb')
        POI_and_E3 = mda.Merge(PROTAC_E3_head.atoms, PROTAC_POI_head.atoms)
        POI_and_E3.atoms.write('./' + PROTAC_folder + '/PROTAC_POI_and_E3_' + str(int(index_list[blockslice.start + ts.frame])) + '.pdb')
        ob_cmd = 'obabel ' + './' + PROTAC_folder + '/PROTAC_POI_and_E3_' + str(int(index_list[blockslice.start + ts.frame])) + '.pdb' + ' -O ' + './' + PROTAC_folder + '/PROTAC_POI_and_E3_' + str(int(index_list[blockslice.start + ts.frame])) + '.sdf'
        print(ob_cmd)
        os.system(ob_cmd)

        template = Chem.SDMolSupplier('./' + PROTAC_folder + '/PROTAC_POI_and_E3_' + str(int(index_list[blockslice.start + ts.frame])) + '.sdf')[0]
        mol = AllChem.MolFromSmiles(PROTAC_smiles)

        try:
            mol = AllChem.ConstrainedEmbed(mol, template, useTethers=False)
            file = open('./' + PROTAC_folder + '/PROTAC_full_' + str(int(index_list[blockslice.start + ts.frame])) + '.sdf','w+')
            file.write(Chem.MolToMolBlock(mol))
            file.close()
            ob_cmd = 'obabel ' + './' + PROTAC_folder + '/PROTAC_full_' + str(int(index_list[blockslice.start + ts.frame])) + '.sdf' + ' -O ' + './' + PROTAC_folder + '/PROTAC_full_' + str(int(index_list[blockslice.start + ts.frame])) + '.pdbqt'
            os.system(ob_cmd)
            box_info_list.append(box_info)
        except:
            print('Adding linker for ' + str(int(index_list[blockslice.start + ts.frame])) + ' failed.')

        PROTAC_E3_head.positions = PROTAC_E3_head_positions_backup
        PROTAC_POI_head.positions = PROTAC_POI_head_positions_backup

    print(str(blockslice) + 'box_info_list:')
    with open('box_info_' + str(blockslice), 'w') as w:
        for box_info in box_info_list:
            w.write(str(box_info[0][0]))
            w.write(str(box_info[1]))
            w.write(str(box_info[2]))
            w.write('\n')



if __name__ == '__main__':
    start_time = time.time()

    alignment_folder = 'alignment_for_DSDP'

    clustered_pdb_list = []
    clustered_pdb_dict = {}
    index_list = []
    for filename in glob.glob(alignment_folder + '/' + '*.pdbqt'):
        #ob_cmd = 'obabel ' + './' + filename + ' -O ' + './' + filename[0:-2]
        #os.system(ob_cmd)

        clustered_pdb_dict[int(filename.split('.')[0].split('_')[-1])] = filename[0:-2]
        index_list.append(int(filename.split('.')[0].split('_')[-1]))

    print("Files and directories in '", alignment_folder, "' :")
    #print(clustered_pdb_dict)

    index_list.sort()
    #print(index_list)
    for i in range(len(index_list)):
        clustered_pdb_list.append(clustered_pdb_dict[index_list[i]])


    #print(clustered_pdb_list)
    print(len(clustered_pdb_list))

    clustered_pdb_list = ['./' + i for i in clustered_pdb_list]
    #print(clustered_pdb_list)
    print(len(clustered_pdb_list))

    #clustered_pdb_list = clustered_pdb_list[0:50]
    #aligned_traj = mda.Universe('traj_rotated.pdb', 'aligned_all.pdb')

    PROTAC_folder = 'PROTACs_for_DSDP'
    if not os.path.exists(PROTAC_folder):
        os.makedirs(PROTAC_folder)
        print(f"Folder '{PROTAC_folder}' created successfully.")
    else:
        print(f"Folder '{PROTAC_folder}' already exists.")


    n_jobs = 35
    n_frames = len(clustered_pdb_list)
    n_blocks = n_jobs
    n_frames_per_block = n_frames // n_blocks
    print('n_frames_per_block:')
    print(n_frames_per_block)

    blocks = [range(i * n_frames_per_block, (i + 1) * n_frames_per_block) for i in range(n_blocks - 1)]
    print('blocks:')
    print(blocks)
    blocks.append(range((n_blocks - 1) * n_frames_per_block, n_frames))
    print('blocks:')
    print(blocks)

    processes = []
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    for bs in blocks:
        p = multiprocessing.Process(target=add_linker_for_PROTAC, args=(bs, clustered_pdb_list, index_list, return_dict, ))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
    print(return_dict)

    end_time = time.time()
    duration = end_time - start_time
    print(f"running time: {duration} s")
