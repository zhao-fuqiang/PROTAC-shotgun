from colabdesign.af.utils import save_pdb_from_aux, cal_iptm
from colabdesign import mk_afdesign_model, clear_mem

import matplotlib.pyplot as plt

class _dock:
    def __init__(self):
        clear_mem()
        self.af_model = mk_afdesign_model(protocol="dock",
                                          recycle_mode="backprop",
                                          crop_len=self.crop_len,
                                          prob_rest=self.prob_rest,
                                          bfloat=self.bfloat,
                                          data_dir=self.data_dir)
        self.af_model.asym_id = self.asym_id
        if self.crop_len is not None:
            self.af_model.sample_p = self.sample_p
            self.af_model.flag_valid = self.flag_valid
        self._empty_records()
    
    def _empty_records(self):
        self.gen_inputs = []
        self.gen_pdb = []
        self.gen_measures = []
        self.dis_measures = []

    def optimize(self, iepoch):
        af_model = self.af_model
        
        # prep_input is the _prep_dock from prep.py
        af_model.prep_inputs(rest_set=self.rest_set,
                             template=self.template,
                             fixed_chains=self.fixed_chains,
                             use_initial=self.use_initial,
                             msas=self.msas, # it's None
                             split_templates=self.split_templates,
                             use_dgram=self.use_dgram,
                             rm_template_seq=self.rm_template_seq)
        af_model.gen_inputs = []
        af_model.gen_outputs = []

        #print('af_model.seq_init from docking.py:')
        #print(af_model.seq_init)
        af_model.restart(seq=af_model.seq_init)
        af_model.set_weights(plddt=0.1, i_pae=0.1, dgram_cce=1.5, 
                             rest_1v1=self.w_res, rest_1vN=self.w_res,
                             rest_MvN=self.w_res, rest_non=self.w_non)
        af_model.opt['weights_bak']['rest'] = self.w_res
        af_model.opt['weights_bak']['rest_non'] = self.w_non
        af_model.opt["rest"]["cutoff"] = self.res_thres
        af_model.opt["rest_non"]["cutoff"] = self.non_thres
        af_model.opt["lr"] = self.lr

        crop = True if self.crop_len is not None else False
        af_model.design(self.step_num, soft=0, temp=1, hard=0,
                        dropout=True, crop=crop,
                        save_every_n_step=self.save_every_n_step)
        
        self.gen_inputs.append(af_model.gen_inputs)
        if self.crop_len is None:
            igen_pdb, igen_measures = [], []
            for i in range(len(af_model.gen_outputs)):
                # save as pdb and template
                i_outputs = af_model.gen_outputs[i]
                save_path = f'{self.save_path}/gen/gen_{iepoch+1}_{i+1}.pdb'
                save_pdb_from_aux(i_outputs, save_path)
                igen_pdb.append({'atom_positions': i_outputs['atom_positions'],
                                 'atom_mask': i_outputs['atom_mask']})
                # save iptm
                iptm = float(cal_iptm(i_outputs['ptm'], self.asym_id))
                plddt = i_outputs['plddt']
                igen_measures.append([iptm, plddt])
            self.gen_pdb.append(igen_pdb)
            self.gen_measures.append(igen_measures)
            #print(str(i + 1) + 'th i_outputs:')
            #print(i_outputs)

        #print('self.gen_inputs:')
        #print(self.gen_inputs)

        '''
        fig, ax = plt.subplots(1, 3, figsize=(10, 50))
        im0 = ax[0].imshow(self.gen_inputs[0][0]['seq']['pseudo'][0], origin='upper')
        im1 = ax[1].imshow(self.gen_inputs[0][1]['seq']['pseudo'][0], origin='upper')
        im2 = ax[2].imshow(self.gen_inputs[0][2]['seq']['pseudo'][0], origin='upper')
        # add residue ID labels to axes
        tick_interval = 20
        plt.title('seq_pseudo')
        # colorbar
        cbar0 = fig.colorbar(im0, ax=ax[0])
        cbar0.ax.set_ylabel('seq_pseudo')
        cbar1 = fig.colorbar(im1, ax=ax[1])
        cbar1.ax.set_ylabel('seq_pseudo')
        cbar2 = fig.colorbar(im2, ax=ax[2])
        cbar2.ax.set_ylabel('seq_pseudo')
        plt.savefig('seq_pseudo.png')

        fig, ax = plt.subplots(1, 3, figsize=(10, 50))
        im0 = ax[0].imshow(self.gen_inputs[0][0]['seq']['pssm'][0], origin='upper')
        im1 = ax[1].imshow(self.gen_inputs[0][1]['seq']['pssm'][0], origin='upper')
        im2 = ax[2].imshow(self.gen_inputs[0][2]['seq']['pssm'][0], origin='upper')
        # add residue ID labels to axes
        tick_interval = 20
        plt.title('seq_pssm')
        # colorbar
        cbar0 = fig.colorbar(im0, ax=ax[0])
        cbar0.ax.set_ylabel('seq_pssm')
        cbar1 = fig.colorbar(im1, ax=ax[1])
        cbar1.ax.set_ylabel('seq_pssm')
        cbar2 = fig.colorbar(im2, ax=ax[2])
        cbar2.ax.set_ylabel('seq_pssm')
        plt.savefig('seq_pssm.png')
        '''

        #print('self.gen_pdb:')
        #print(self.gen_pdb)

        '''
        fig, ax = plt.subplots(1, 3, figsize=(10, 50))
        im0 = ax[0].imshow(self.gen_pdb[0][0]['atom_mask'], origin='upper')
        im1 = ax[1].imshow(self.gen_pdb[0][1]['atom_mask'], origin='upper')
        im2 = ax[2].imshow(self.gen_pdb[0][2]['atom_mask'], origin='upper')
        # add residue ID labels to axes
        tick_interval = 20
        plt.title('atom_mask')
        # colorbar
        cbar0 = fig.colorbar(im0, ax=ax[0])
        cbar0.ax.set_ylabel('atom_mask')
        cbar1 = fig.colorbar(im1, ax=ax[1])
        cbar1.ax.set_ylabel('atom_mask')
        cbar2 = fig.colorbar(im2, ax=ax[2])
        cbar2.ax.set_ylabel('atom_mask')
        plt.savefig('atom_mask.png')
        '''

    
    def inference(self):
        af_model = self.af_model
        # if use crop mode, templates should be first generated
        if self.crop_len is not None:
            print('generation stage inference:')
            for iepoch in range(len(self.gen_inputs)):
                print(f'infer epoch {iepoch+1}')
                af_model.gen_inputs = self.gen_inputs[iepoch]
                save_path = f'{self.save_path}/gen/gen_{iepoch+1}'
                af_model.gen_infer(save_path)
                self.gen_pdb.append(af_model.gen_pdb)
                self.gen_measures.append(af_model.gen_measures)
        
        print('prediction stage inference:')
        for iepoch in range(len(self.gen_inputs)):
            print(f'infer epoch {iepoch+1}')
            af_model.gen_inputs = self.gen_inputs[iepoch]
            af_model.gen_outputs = self.gen_pdb[iepoch]
            save_path = f'{self.save_path}/pred/pred_{iepoch+1}'
            af_model.dis_infer(save_path)
            self.dis_measures.append(af_model.dis_measures)

    def PROTAC_inference(self, PD):
        af_model = self.af_model

        PROTAC_gen_pdb = {}
        PROTAC_gen_pdb['atom_positions'] = PD.prot_obj['batch']['all_atom_positions']
        PROTAC_gen_pdb['atom_mask'] = PD.prot_obj['batch']['all_atom_mask']
        
        print('PROTAC prediction stage inference:')
        for iepoch in range(len(self.gen_inputs)):
            print(f'infer epoch {iepoch+1}')
            af_model.gen_inputs = self.gen_inputs[iepoch]
            af_model.gen_outputs = [PROTAC_gen_pdb]
            save_path = f'{self.save_path}/pred/pred_{iepoch+1}'
            af_model.dis_infer(save_path)
            self.dis_measures.append(af_model.dis_measures)
