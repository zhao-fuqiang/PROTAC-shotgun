import numpy as np
import os, sys
import copy

from colabdock.utils import prep_path
from colabdock.model import ColabDock

from colabdesign.af.prep import prep_pdb
from colabdesign.af.alphafold.common import residue_constants

from colabdock.docking import _dock
from colabdock.ranking import _rank
from colabdock.prep import _rest

from config import config

class PROTAC_Dock():
    def __init__(self,
                 template,
                 restraints,
                 save_path,
                 data_dir,
                 msa_path=None,
                 structure_gt=None,
                 crop_len=None,
                 fixed_chains=None,
                 round_num=2,
                 step_num=50,
                 prob_rest=0.5,
                 bfloat=True,
                 res_thres=8.0,
                 non_thres=12.0,
                 save_every_n_step=1) -> None:
        self.template = template
        self.structure_gt = structure_gt
        self.fixed_chains = fixed_chains

        self.rest_raw = restraints
        self.res_thres = res_thres
        self.non_thres = non_thres

        self.step_num = step_num
        self.round_num = round_num
        self.crop_len = crop_len
        self.prob_rest = prob_rest
        self.bfloat = bfloat

        self.save_path = save_path
        self.data_dir = data_dir
        self.save_every_n_step = save_every_n_step

        self.use_initial = True
        self.split_templates = False
        self.use_dgram = True
        self.msas = msa_path
        self.rm_template_seq = False

        self.w_non = 1.0
        self.w_res = 2.0
        self.lr = 0.1


    def setup(self):
        self.prot_obj = prep_pdb(self.template['pdb_path'],
                                 chain=self.template['chains'],
                                 for_alphafold=False)
        
        #print('self.prot_obj in setup()')
        #print(self.prot_obj)

        self.seq_wt = ''.join([residue_constants.restypes[ind] for ind in self.prot_obj['batch']['aatype']])



if __name__ == '__main__':
    #print(config)
    template = {'pdb_path': config.template,
                'chains': config.chains}
    PD = PROTAC_Dock(template,
                    [],
                    config.save_path,
                    config.data_dir)
    PD.setup()

    #print('PD.prot_obj after setup()')
    #print(PD.prot_obj['residue_index'])
    #print(PD.prot_obj['idx'])

    print('config.rest_MvN:')
    print(config.rest_MvN)

    chains = [None] if config.chains is None else config.chains.split(",")
    rest_pockets = copy.deepcopy(config.rest_pockets)
    if config.rest_pockets is not None:
        pocket_i = 0
        for reslist_pocket in config.rest_pockets:
            index_start = np.where(PD.prot_obj['idx']['chain'] == chains[pocket_i])[0][0]

            res_pocket_i = 0
            for resindex_pocket in reslist_pocket:
                resindex = PD.prot_obj['idx']['residue'].tolist().index(resindex_pocket, index_start) + 1
                rest_pockets[pocket_i][res_pocket_i] = resindex

                res_pocket_i += 1
            pocket_i += 1

        rest_MvN_list = []
        pocket_i = 0
        for reslist_pocket in rest_pockets:
            rest_MvN = []
            for resindex_pocket in reslist_pocket:
                rest_1vN = [resindex_pocket]
                if pocket_i == 0:
                    i = 1
                elif pocket_i == 1:
                    i = 0
                else:
                    i = 100
                rest_1vN.append(rest_pockets[i])
                rest_MvN.append(rest_1vN)
            rest_MvN.append(config.rest_pockets_n)
            rest_MvN_list.append(rest_MvN)
            pocket_i += 1

    else:
        rest_MvN = None

    config.rest_MvN = rest_MvN_list
    #print('config.rest_pockets_n:')
    #print(config.rest_pockets_n)
    #print('config.rest_pockets:')
    #print(config.rest_pockets)
    #print('rest_pockets:')
    #print(rest_pockets)

    save_path = config.save_path
    prep_path(save_path)
    ######################################################################################
    # template and native structure
    ######################################################################################
    template_r = config.template
    native_r = config.native
    chains = config.chains
    template = {'pdb_path': template_r,
                'chains': chains}
    native = {'pdb_path': native_r,
              'chains': chains}
    fixed_chains = config.fixed_chains

    ######################################################################################
    # experimental restraints
    ######################################################################################
    hahaha = True
    if hahaha:
        rest_MvN_r = config.rest_MvN
        rest_non_r = config.rest_rep
        rest_1vN_r = config.rest_1vN
        rest_1v1_r = config.rest_1v1

        # 1v1
        if rest_1v1_r is not None:
            if type(rest_1v1_r[0]) is not list:
                rest_1v1_r = [rest_1v1_r]
            rest_1v1 = np.array(rest_1v1_r) - 1
        else:
            rest_1v1 = None

        # 1vN
        if rest_1vN_r is not None:
            if type(rest_1vN_r[0]) is not list:
                rest_1vN_r = [rest_1vN_r]
            rest_1vN = []
            for irest_1vN in rest_1vN_r:
                rest_1vN.append([irest_1vN[0] - 1, np.array(irest_1vN[1]) - 1])
        else:
            rest_1vN = None

        # MvN
        if rest_MvN_r is not None:
            if type(rest_MvN_r[-1]) is not list:
                rest_MvN_r = [rest_MvN_r]
            rest_MvN = []
            for irest_MvN in rest_MvN_r:
                irest = []
                for irest_1vN in irest_MvN[:-1]:
                    irest.append([irest_1vN[0] - 1, np.array(irest_1vN[1]) - 1])
                irest.append(irest_MvN[-1])
                rest_MvN.append(irest)
        else:
            rest_MvN = None

        # repulsive
        if rest_non_r is not None:
            if type(rest_non_r[0]) is not list:
                rest_non_r = [rest_non_r]
            rest_non = np.array(rest_non_r) - 1
        else:
            rest_non = None

        restraints = {'1v1': rest_1v1,
                      '1vN': rest_1vN,
                      'MvN': rest_MvN,
                      'non': rest_non}

        res_thres = config.res_thres
        non_thres = config.rep_thres

    print('rest_MvN:')
    print(rest_MvN)

    ######################################################################################
    # optimization parameters
    ######################################################################################
    rounds = config.rounds
    crop_len = config.crop_len
    step_num = config.steps
    save_every_n_step = config.save_every_n_step
    data_dir = config.data_dir
    bfloat = config.bfloat

    ######################################################################################
    # start docking
    ######################################################################################
    dock_model = ColabDock(template,
                           restraints,
                           save_path,
                           data_dir,
                           structure_gt=native,
                           crop_len=crop_len,
                           fixed_chains=fixed_chains,
                           round_num=rounds,
                           step_num=step_num,
                           bfloat=bfloat,
                           res_thres=res_thres,
                           non_thres=non_thres,
                           save_every_n_step=save_every_n_step)
    dock_model.setup()

    print('\nStart optimization')


    for ith in range(rounds):
        dock_model.optimize(ith)
        #break
    #print('self.prot_obj after setup()')
    #print(PD.prot_obj)
    
    #dock_model.PROTAC_inference(PD)
    dock_model.inference()

