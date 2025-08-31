
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


def cal_PPI_CV_v1(blockslice, clustered_pdb_list, traj_E3_sel, traj_POI_sel, return_dict):

    alignment_folder = 'alignment_for_DSDP'
    if not os.path.exists(alignment_folder):
        os.makedirs(alignment_folder)

    print('cal_PPI_CV_v1:')
    print(blockslice)
    print(blockslice.start)
    print(blockslice.stop)

    clustered_traj = mda.Universe(clustered_pdb_list[0], clustered_pdb_list[blockslice.start:blockslice.stop])
    print(clustered_pdb_list[blockslice.start:blockslice.stop])
    print(len(clustered_traj.trajectory))

    ref_for_traj = mda.Universe('8QW7_AB.pdb', '8QW7_AB.pdb')
    ref_for_E3 = mda.Universe('8qw7_B.pdb', '8qw7_B.pdb')

    traj_E3 = clustered_traj.select_atoms(traj_E3_sel)
    print(len(traj_E3.atoms))
    traj_POI = clustered_traj.select_atoms(traj_POI_sel)
    print(len(traj_POI.atoms))

    # E3 is receptor, and POI is ligand.
    ref_for_E3_com = ref_for_E3.atoms.center(weights=None)
    print(ref_for_E3_com)
    ref_E3_CA = ref_for_E3.select_atoms('name CA')
    print(len(ref_E3_CA.atoms))

    print(ref_E3_CA.positions[0:3,:])
    ref_for_E3.atoms.translate(-ref_for_E3_com)
    print(ref_E3_CA.positions[0:3,:])
    #ref_for_E3.atoms.write("E3_ref_translated.pdb")

    traj_E3_CA = clustered_traj.select_atoms('name CA and chainID B')
    print(len(traj_E3_CA.atoms))



    results = []

    with mda.Writer('aligned_all.pdb', clustered_traj.atoms.n_atoms) as w:
        for ts in clustered_traj.trajectory:
            print(blockslice)
            print(ts.frame)
            r = np.linalg.norm(traj_E3.positions[0] - traj_POI.positions[0])
            results.append(r)

            traj_POI_COM = traj_POI.center_of_mass()
            traj_POI_posi_backup = traj_POI.positions
            traj_E3_COM = traj_E3.center_of_mass()
            traj_E3_posi_backup = traj_E3.positions
            for i in range(10):
                angle_perturbation = np.random.random(3) * 10
                rotation_perturbation = Rotation.from_euler("zyx", angle_perturbation, degrees=True)
                trans_perturbation = np.random.random(3) * 2

                #clustered_traj.atoms.write("traj_orig.pdb")
                traj_POI.atoms.translate(-traj_POI_COM)
                #clustered_traj.atoms.write("traj_pert0.pdb")
                traj_POI.atoms.rotate(rotation_perturbation.as_matrix())
                #clustered_traj.atoms.write("traj_pert1.pdb")
                traj_POI.atoms.translate(traj_POI_COM)
                #clustered_traj.atoms.write("traj_pert2.pdb")
                traj_POI.atoms.translate(trans_perturbation)
                #clustered_traj.atoms.write("traj_pert3.pdb")

                traj_E3_CA_com = traj_E3_CA.atoms.center(weights=None)
                clustered_traj.atoms.translate(-traj_E3_CA_com)
                if ts.frame == 0:
                    clustered_traj.atoms.write('traj_translated.pdb')

                R_E3 = align.rotation_matrix(traj_E3_CA.positions,
                                             ref_E3_CA.positions)[0]
                print(R_E3)

                clustered_traj.atoms.rotate(R_E3)
                if ts.frame == 0:
                    clustered_traj.atoms.write('traj_rotated.pdb')

                clustered_traj.atoms.translate(ref_for_E3_com)
                #w.write(clustered_traj.atoms)


                clustered_traj.atoms.write('Colab_aligned_' + str(blockslice.start) + '.pdb')
                cp_cmd = 'cp ' + 'Colab_aligned_' + str(blockslice.start) + '.pdb' + ' ' + './' + alignment_folder + '/Colab_aligned_' + str(int((blockslice.start + ts.frame) * 10 + i)) + '.pdb '
                os.system(cp_cmd)
                ob_cmd = 'obabel ' + 'Colab_aligned_' + str(blockslice.start) + '.pdb' + ' -O ' + './' + alignment_folder + '/Colab_aligned_' + str(int((blockslice.start + ts.frame) * 10 + i)) + '.pdbqt ' + '-xr'
                os.system(ob_cmd)

                traj_POI.positions = traj_POI_posi_backup
                traj_E3.positions = traj_E3_posi_backup


        return_dict[blockslice.start] = results

    return results


if __name__ == '__main__':
    start_time = time.time()

    clustered_pdb_list = []
    for iter in range(0,100,1):
        path = 'PROTAC_8QW7_results/FFT_cluster_' + str(iter) + '/pred'

        for filename in glob.glob(path + '/' + '*.pdb'):
            clustered_pdb_list.append(filename)

    #print(clustered_pdb_list)
    #print(len(clustered_pdb_list))

    clustered_pdb_list = ['./' + i for i in clustered_pdb_list]
    #print(clustered_pdb_list)
    #print(len(clustered_pdb_list))

    #clustered_pdb_list = clustered_pdb_list[0:50]


    n_jobs = 25
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

    traj_E3_sel = 'chainID B'
    traj_POI_sel = 'chainID A'


    para = 1
    processes = []
    results = []
    queue = multiprocessing.Queue()
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    if para == 1:
        for bs in blocks:
            p = multiprocessing.Process(target=cal_PPI_CV_v1, args=(bs, clustered_pdb_list, traj_E3_sel, traj_POI_sel, return_dict, ))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        print(return_dict)
    else:
        for bs in blocks:
            p = cal_PPI_CV_v1(bs, clustered_pdb_list, traj_E3_sel, traj_POI_sel, return_dict)
            print('p:')
            print(p)
            results.append(p)

    print('results:')
    print(results)

    end_time = time.time()
    duration = end_time - start_time
    print(f"running time: {duration} s")

