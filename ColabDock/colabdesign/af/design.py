import random
import jax
import jax.numpy as jnp
import numpy as np
import joblib
from colabdesign.shared.utils import copy_dict, Key, dict_to_str, to_float
from colabdesign.af.utils import save_pdb_from_aux

import matplotlib.pyplot as plt

try:
  from jax.example_libraries.optimizers import sgd, adam
except:
  from jax.experimental.optimizers import sgd, adam

####################################################
# AF_DESIGN - design functions
####################################################
#\
# \_af_design
# |\
# | \_restart
#  \
#   \_design
#    \_step
#     \_run
#      \_recycle
#       \_single
#
####################################################

class _af_design:

  def restart(self, seed=None, optimizer="sgd", opt=None, weights=None,
              seq=None, keep_history=False, reset_opt=True, **kwargs):   
    '''
    restart the optimization
    ------------
    note: model.restart() resets the [opt]ions and weights to their defaults
    use model.set_opt(..., set_defaults=True) and model.set_weights(..., set_defaults=True)
    or model.restart(reset_opt=False) to avoid this
    ------------
    seed=0 - set seed for reproducibility
    reset_opt=False - do NOT reset [opt]ions/weights to defaults
    keep_history=True - do NOT clear the trajectory/[opt]ions/weights
    '''
    # reset [opt]ions
    if reset_opt and not keep_history:
      self.opt = copy_dict(self._opt)

    # update options/settings (if defined)
    self.set_opt(opt)
    self.set_weights(weights)

    #print('self.params from restart() in design.py:')
    #print(self.params)
    # initialize sequence
    self.key = Key(seed=seed).get
    self.set_seq(seq=seq, **kwargs)

    # setup optimizer
    if isinstance(optimizer, str):
      # if optimizer == "adam": (optimizer,lr) = (adam,0.02)
      # if optimizer == "sgd":  (optimizer,lr) = (sgd,0.1)
      if optimizer == "adam": optimizer = adam
      if optimizer == "sgd":  optimizer = sgd

    self._init_fun, self._update_fun, self._get_params = optimizer(1.0)
    self._state = self._init_fun(self.params)
    self._k = 0
    #print('self.params from restart() in design.py:')
    #print(self.params)
    #print('self._state from restart() in design.py:')
    #print(self._state)

    if not keep_history:
      # initialize trajectory
      self._traj = {"log":[],"seq":[],"xyz":[],"plddt":[],"pae":[]}
      self._best_metric = self._best_loss = np.inf
      self._best_aux = self._best_outs = None

    # set crop length
    if self._args["crop"] is False:
      self._args["crop_len"] = self._inputs["residue_index"].shape[-1]
    else:
      # adjust the sampling prob according to the loss
      self.loss_history = [None, None, None, None]
      
  def run(self, model=None, backprop=True, crop=True, average=True, callback=None):
    '''run model to get outputs, losses and gradients'''

    # crop inputs 
    (L, max_L) = (self._inputs["residue_index"].shape[-1], self._args["crop_len"])
    mask_1v1 = self.rest_set['rest_mask']['1v1']
    mask_1vN = self.rest_set['rest_mask']['1vN']
    mask_MvN = self.rest_set['rest_mask']['MvN']
    mask_non = self.rest_set['rest_mask']['non']
    if crop and max_L < L:
      # pick randomly according to contacts and restraints
      if hasattr(self, "lens") and len(self.lens) > 1:
        # multi-chains in protein
        mask_1v1_ = None if mask_1v1 is None else np.zeros([L, L])
        mask_1vN_ = None if mask_1vN is None else np.zeros([L, L])
        mask_MvN_ = None if mask_MvN is None else np.zeros([1, L, L])
        mask_non_ = None if mask_non is None else np.zeros([L, L])
        masks = {'1v1': mask_1v1_,
                 '1vN': mask_1vN_,
                 'MvN': mask_MvN_,
                 'non': mask_non_}
        self.opt['num_MvN'] = [1]
        self.set_weights(rest_1v1=0.0, rest_1vN=0.0, rest_MvN=0.0, rest_non=0.0)

        # which type of restraints or two segs in two proteins are optimized
        rand_num = random.random()
        if rand_num < self.sample_p[0]:
          # optimize 1v1
          p = self.rest_set['rest_p']['1v1']
          mask = self.rest_set['rest_mask']['1v1']
          masks['1v1'] = mask
          self.set_weights(rest_1v1=self.opt['weights_bak']['rest'])
        elif rand_num < self.sample_p[1]:
          # optimize 1vN
          p = self.rest_set['rest_p']['1vN']
          mask = self.rest_set['rest_mask']['1vN']
          masks['1vN'] = mask
          self.set_weights(rest_1vN=self.opt['weights_bak']['rest'])
        elif rand_num < self.sample_p[2]:
          # optimize MvN
          ps = self.rest_set['rest_p']['MvN']
          MvN_ind = random.choice(np.arange(len(ps)))
          p = ps[MvN_ind]
          num_MvN = self.rest_set['rest_MvN_num'][MvN_ind]
          self.opt['num_MvN'] = [num_MvN]
          mask = self.rest_set['rest_mask']['MvN'][MvN_ind]
          masks['MvN'] = mask[None]
          self.set_weights(rest_MvN=self.opt['weights_bak']['rest'])
        elif rand_num < self.sample_p[3]:
          # optimize non
          p = self.rest_set['rest_p']['non']
          mask = self.rest_set['rest_mask']['non']
          masks['non'] = mask
          self.set_weights(rest_non=self.opt['weights_bak']['rest_non'])
        else:
          start = random.choice(np.arange(L-max_L))
          p = np.arange(start, start+max_L)
    else:
      p = np.arange(L)
      masks = {'1v1': mask_1v1,
               '1vN': mask_1vN,
               'MvN': mask_MvN,
               'non': mask_non}
      self.opt['num_MvN'] = self.rest_set['rest_MvN_num']
    
    self.opt["crop_pos"] = p
    for k, v in masks.items():
      if v is not None:
        masks[k] = jnp.array(v, dtype=jnp.float32)
    self._batch["masks_rest"] = masks
    
    # decide which model params to use
    ns,ns_name = [],[]
    for n,name in enumerate(self._model_names):
      if "openfold" in name:
        if self._args["use_openfold"]:  ns.append(n); ns_name.append(name)
      elif self._args["use_alphafold"]: ns.append(n); ns_name.append(name)

    # sub select number of model params
    if model is not None:
      model = [model] if isinstance(model,int) else list(model)
      ns = [ns[n if isinstance(n,int) else ns_name.index(n)] for n in model]
    
    ns = jnp.array(ns)
    m = min(self.opt["models"],len(ns))
    if self.opt["sample_models"] and m != len(ns):
      model_num = jax.random.choice(self.key(),ns,(m,),replace=False)
    else:
      model_num = ns[:m]      
    model_num = np.array(model_num).tolist()

    # loop through model params
    outs = []
    for n in model_num:
      p = self._model_params[n]
      outs.append(self._recycle(p, backprop=backprop))
    outs = jax.tree_map(lambda *x: jnp.stack(x), *outs)

    if average:
      # update gradients
      self.grad = jax.tree_map(lambda x: x.mean(0), outs["grad"])

      # update [aux]iliary outputs
      self.aux = jax.tree_map(lambda x:x[0], outs["aux"])

      # update loss (take mean across models)
      self.loss = outs["loss"].mean()      
      self.aux["losses"] = jax.tree_map(lambda x: x.mean(0), outs["aux"]["losses"])

    else:
      self.loss, self.aux, self.grad = outs["loss"], outs["aux"], outs["grad"]

    # callback
    if callback is not None: callback(self)

    # update log
    self.aux["log"] = copy_dict(self.aux["losses"])
    self.aux["log"].update({"hard": self.opt["hard"], "soft": self.opt["soft"],
                            "temp": self.opt["temp"], "loss": self.loss,
                            "recycles": self.aux["recycles"]})

    # compute sequence recovery
    if self.protocol in ["fixbb","partial"] or (self.protocol == "binder" and self._args["redesign"]):
      aatype = self.aux["seq"]["pseudo"].argmax(-1)
      if self.protocol == "partial" and "pos" in self.opt:
        aatype = aatype[...,self.opt["pos"]]
      self.aux["log"]["seqid"] = (aatype == self._wt_aatype).mean()

    self.aux["log"] = to_float(self.aux["log"])
    self.aux["log"]["models"] = model_num
    
    # backward compatibility
    self._outs = self.aux
    #print(dir(self))

    #print(self.aux['atom_positions'].shape)
    #print(self.aux['atom_positions'])
    #np.savetxt('aux_atom_positions_' + str(self._k) + '_0.out', self.aux['atom_positions'][0], delimiter=',')
    #np.savetxt('aux_atom_positions_' + str(self._k) + '_1.out', self.aux['atom_positions'][1], delimiter=',')
    #np.savetxt('aux_atom_positions_' + str(self._k) + '_2.out', self.aux['atom_positions'][2], delimiter=',')

  def _single(self, model_params, backprop=True):
    '''single pass through the model'''
    flags  = [self.params, model_params, self._inputs, self._batch, self.key(), self.opt]
    #print('self.params:')
    #print(self.params['seq'].shape)
    #print(self.params)
    #print(model_params)
    #print('self._inputs.keys:')
    #print(self._inputs.keys())
    #print('self._inputs_target_feat:')
    #print(self._inputs['target_feat'].shape)
    #print(self._inputs['target_feat'])
    #print(self._inputs)
    #print('self._inputs_prev_keys:')
    #print(self._inputs['prev'].keys())
    #print('self._inputs_prev_prev_msa_first_row:')
    #print(self._inputs['prev']['prev_msa_first_row'].shape)
    #print(self._inputs['prev']['prev_msa_first_row'])
    #print('self._inputs_prev_prev_pair:')
    #print(self._inputs['prev']['prev_pair'].shape)
    #print(self._inputs['prev']['prev_pair'])
    #print('self._inputs_prev_prev_pos:')
    #print(self._inputs['prev']['prev_pos'].shape)
    #print(self._inputs['prev']['prev_pos'])
    #print('self._batch.keys:')
    #print(self._batch.keys())
    #print('rigidgroups_alt_gt_frames:')
    #print(self._batch['rigidgroups_alt_gt_frames'].shape)
    ##print(self._batch['rigidgroups_alt_gt_frames'])
    #print('rigidgroups_group_exists:')
    #print(self._batch['rigidgroups_group_exists'].shape)
    ##print(self._batch['rigidgroups_group_exists'])
    #print('rigidgroups_group_is_ambiguous:')
    #print(self._batch['rigidgroups_group_is_ambiguous'].shape)
    ##print(self._batch['rigidgroups_group_is_ambiguous'])
    #print('rigidgroups_gt_exists:')
    #print(self._batch['rigidgroups_gt_exists'].shape)
    ##print(self._batch['rigidgroups_gt_exists'])
    #print('rigidgroups_gt_frames:')
    #print(self._batch['rigidgroups_gt_frames'].shape)
    ##print(self._batch['rigidgroups_gt_frames'])
    #print('mask_d:')
    #print(self._batch['mask_d'].shape)
    #print(self._batch['mask_d'])
    #print('mask_dgram:')
    #print(self._batch['mask_dgram'].shape)
    #print(self._batch['mask_dgram'])
    #print('masks_rest:MvN:')
    #print(self._batch['masks_rest']['MvN'].shape)
    ##print(self._batch['masks_rest']['MvN'])
    #print('masks_rest:non:')
    #print(self._batch['masks_rest']['non'].shape)
    ##print(self._batch['masks_rest']['non'])
    '''
    fig, ax = plt.subplots(4, 1, figsize=(50, 50))
    im0 = ax[0].imshow(np.transpose(self._inputs['prev']['prev_msa_first_row']), origin='upper')
    im1 = ax[1].imshow(np.transpose(self._inputs['prev']['prev_pair'][self._k * 3]), origin='upper')
    im2 = ax[2].imshow(np.transpose(self._inputs['prev']['prev_pair'][self._k * 3 + 1]), origin='upper')
    im3 = ax[3].imshow(np.transpose(self._inputs['prev']['prev_pair'][self._k * 3 + 2]), origin='upper')
    tick_interval = 20
    # colorbar
    cbar0 = fig.colorbar(im0, ax=ax[0])
    cbar0.ax.set_ylabel('prev_msa_first_row')
    cbar1 = fig.colorbar(im1, ax=ax[1])
    cbar1.ax.set_ylabel('prev_pair:' + str(self._k * 3))
    cbar2 = fig.colorbar(im2, ax=ax[2])
    cbar2.ax.set_ylabel('prev_pair:' + str(self._k * 3 + 1))
    cbar3 = fig.colorbar(im3, ax=ax[3])
    cbar3.ax.set_ylabel('prev_pair:' + str(self._k * 3 + 2))
    plt.savefig('msa_feat_and_prev_' + str(self._k) + '.png')

    fig, ax = plt.subplots(2, 1, figsize=(200, 50))
    im0 = ax[0].imshow(np.transpose(self._inputs['target_feat']), origin='upper')
    tick_interval = 20
    # colorbar
    cbar0 = fig.colorbar(im0, ax=ax[0])
    cbar0.ax.set_ylabel('self._inputs_target_feat')
    plt.savefig('self._inputs_target_feat_' + str(self._k) + '.png')


    fig, ax = plt.subplots(3, 3, figsize=(50, 50))
    im00 = ax[0][0].imshow(np.transpose(self._batch['mask_d']), origin='upper')
    im01 = ax[0][1].imshow(np.transpose(self._batch['mask_dgram']), origin='upper')
    im02 = ax[0][2].imshow(np.transpose(self._batch['masks_rest']['non']), origin='upper')
    im10 = ax[1][0].imshow(np.transpose(self._batch['masks_rest']['MvN'][0]), origin='upper')
    im11 = ax[1][1].imshow(np.transpose(self._batch['masks_rest']['MvN'][1]), origin='upper')
    im12 = ax[1][2].imshow(np.transpose(self._batch['masks_rest']['MvN'][2]), origin='upper')
    im20 = ax[2][0].imshow(np.transpose(self._batch['rigidgroups_alt_gt_frames'][self._k * 3]), origin='upper')
    im21 = ax[2][1].imshow(np.transpose(self._batch['rigidgroups_alt_gt_frames'][self._k * 3 + 1]), origin='upper')
    im22 = ax[2][2].imshow(np.transpose(self._batch['rigidgroups_alt_gt_frames'][self._k * 3 + 2]), origin='upper')
    tick_interval = 20
    # colorbar
    cbar00 = fig.colorbar(im00, ax=ax[0][0])
    cbar00.ax.set_ylabel('mask_d')
    cbar01 = fig.colorbar(im01, ax=ax[0][1])
    cbar01.ax.set_ylabel('mask_dgram')
    cbar02 = fig.colorbar(im02, ax=ax[0][2])
    cbar02.ax.set_ylabel('masks_rest:non')
    cbar10 = fig.colorbar(im10, ax=ax[1][0])
    cbar10.ax.set_ylabel('masks_rest:MvN')
    cbar11 = fig.colorbar(im11, ax=ax[1][1])
    cbar11.ax.set_ylabel('masks_rest:MvN')
    cbar12 = fig.colorbar(im12, ax=ax[1][2])
    cbar12.ax.set_ylabel('masks_rest:MvN')
    cbar20 = fig.colorbar(im20, ax=ax[2][0])
    cbar20.ax.set_ylabel('rigidgroups_alt_gt_frames:' + str(self._k * 3))
    cbar21 = fig.colorbar(im21, ax=ax[2][1])
    cbar21.ax.set_ylabel('rigidgroups_alt_gt_frames:' + str(self._k * 3 + 1))
    cbar22 = fig.colorbar(im22, ax=ax[2][2])
    cbar22.ax.set_ylabel('rigidgroups_alt_gt_frames:' + str(self._k * 3 + 2))
    plt.savefig('self._batch' + str(self._k) + '.png')
    '''
    #print('self.key:')
    #print(self.key())
    #print('self.opt:')
    #print(self.opt)
    if backprop:
      (loss, aux), grad = self._grad_fn(*flags)
      print('backprop is on')
    else:
      loss, aux = self._fn(*flags)
      grad = jax.tree_map(jnp.zeros_like, self.params)
      print('backprop is off')

    pdb_initial_dict = {"aatype": self._inputs['aatype'],
                        "residue_index": self._inputs['residue_index'],
                        "atom_positions": self._inputs['prev']['prev_pos'],
                        "atom_mask": aux['atom_mask'],
                        "plddt": aux['plddt']}
    #save_pdb_from_aux(pdb_initial_dict, filename='pdb_initial_' + str(self._k) + '.pdb')

    #print('aux.keys:')
    #print(aux.keys())
    #print(aux)
    #print("aux['losses']:")
    #print(aux['losses'])
    #print('loss:')
    #print(loss)
    #print("grad['seq']:")
    #print(grad['seq'].shape)
    #print(grad['seq'][0])
    '''
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(200, 50))
    im0 = ax[0].imshow(np.transpose(grad['seq'][0]), origin='upper')
    im1 = ax[1].imshow(np.transpose(aux['atom_mask']), origin='upper')
    im2 = ax[2].imshow(np.transpose(self.params['seq'][0]), origin='upper')
    # add residue ID labels to axes
    tick_interval = 20
    plt.ylabel('')
    plt.xlabel('')
    plt.title('grad')
    # colorbar
    cbar0 = fig.colorbar(im0, ax=ax[0])
    cbar0.ax.set_ylabel('grad ()')
    cbar1 = fig.colorbar(im1, ax=ax[1])
    cbar1.ax.set_ylabel('atom_mask')
    cbar2 = fig.colorbar(im2, ax=ax[2])
    cbar2.ax.set_ylabel('self.params')
    plt.savefig('grad_after_grad_fn' + str(self._k) + '.png')


    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(200, 10))
    im0 = ax[0].imshow(np.transpose(self._batch['rigidgroups_group_exists']), origin='upper')
    im1 = ax[1].imshow(np.transpose(self._batch['rigidgroups_group_is_ambiguous']), origin='upper')
    im2 = ax[2].imshow(np.transpose(self._batch['rigidgroups_gt_exists']), origin='upper')
    tick_interval = 20
    # colorbar
    cbar0 = fig.colorbar(im0, ax=ax[0])
    cbar0.ax.set_ylabel('rigidgroups_group_exists')
    cbar1 = fig.colorbar(im1, ax=ax[1])
    cbar1.ax.set_ylabel('rigidgroups_group_is_ambiguous')
    cbar2 = fig.colorbar(im2, ax=ax[2])
    cbar2.ax.set_ylabel('rigidgroups_gt_exists')
    plt.savefig('rigidgroups' + str(self._k) + '.png')
    '''
    return {"loss":loss, "aux":aux, "grad":grad}

  def _recycle(self, model_params, backprop=True):   
    '''multiple passes through the model (aka recycle)'''

    mode = self._args["recycle_mode"]
    if mode in ["backprop","add_prev"]:
      recycles = self.opt["recycles"] = self._runner.config.model.num_recycle
      out = self._single(model_params, backprop)
      print('design, recycle, mode 1')
    
    else:
      recycles = self.opt["recycles"]
      if mode == "average":
        if "crop_pos" in self.opt: L = self.opt["crop_pos"].shape[0]
        else: L = self._inputs["residue_index"].shape[-1]
        self._inputs["prev"] = {'prev_msa_first_row': np.zeros([L,256]),
                                'prev_pair': np.zeros([L,L,128]),
                                'prev_pos': np.zeros([L,37,3])}
        grad = []
        for _ in range(recycles+1):
          out = self._single(model_params, backprop)
          grad.append(out["grad"])
          self._inputs["prev"] = out["aux"]["prev"]
        out["grad"] = jax.tree_map(lambda *x: jnp.stack(x).mean(0), *grad)
        print('design, recycle, mode 2')
      
      elif mode == "sample":
        self.set_opt(recycles=jax.random.randint(self.key(),[],0,recycles+1))
        out = self._single(model_params, backprop)
        (self.opt["recycles"],recycles) = (recycles,self.opt["recycles"])
        print('design, recycle, mode 3')
      
      else:
        out = self._single(model_params, backprop)
        print('design, recycle, mode 4')
    
    out["aux"]["recycles"] = recycles
    return out

  def step(self, lr_scale=1.0, model=None, backprop=True, crop=True,
           callback=None, save_best=False, verbose=1, grad_clip=False,
           clip_range=0.2, pseudo_loss=0.2, step_iter=0):
    '''do one step of gradient descent'''
    
    self._update_template_mask(self._inputs, self.opt, self.key(), step_iter)

    # run
    self.run(model=model, backprop=backprop, crop=crop, callback=callback)

    # normalize gradient
    tmp = jnp.linalg.norm(self.grad["seq"])
    if grad_clip and tmp > 3:
      self.grad["seq"] = jnp.clip(self.grad["seq"], a_min=-clip_range, a_max=clip_range)

    g = self.grad["seq"]
    gn = jnp.linalg.norm(g,axis=(-1,-2),keepdims=True)
    
    eff_len = (jnp.square(g).sum(-1,keepdims=True) > 0).sum(-2,keepdims=True)
    self.grad["seq"] *= jnp.sqrt(eff_len)/(gn+1e-7)

    # set learning rate
    lr = self.opt["lr"] * lr_scale
    self.grad = jax.tree_map(lambda x:x*lr, self.grad)

    #fig, ax = plt.subplots(3, 1, sharex=True, figsize=(100, 20))

    #print('self._k:')
    #print(self._k)
    #print('self.grad:')
    #print(self.grad)
    #print('self._state:')
    #print(self._state)

    #im0 = ax[0].imshow(np.transpose(self.params['seq'][0]), origin='upper')
    #im1 = ax[1].imshow(np.transpose(self.grad['seq'][0]), origin='upper')

    # apply gradient
    self._state = self._update_fun(self._k, self.grad, self._state)
    self.params = self._get_params(self._state)

    #print('self._state:')
    #print(self._state)
    #print('self.params:')
    #print(self.params)

    #im2 = ax[2].imshow(np.transpose(self.params['seq'][0]), origin='upper')
    #tick_interval = 20
    # colorbar
    #cbar0 = fig.colorbar(im0, ax=ax[0])
    #cbar0.ax.set_ylabel('self.params')
    #cbar1 = fig.colorbar(im1, ax=ax[1])
    #cbar1.ax.set_ylabel('self.grad')
    #cbar2 = fig.colorbar(im2, ax=ax[2])
    #cbar2.ax.set_ylabel('self.params')
    #plt.savefig('apply_gradient' + str(self._k) + '.png')

    # increment
    self._k += 1

    # save results
    self.save_results(save_best=save_best, verbose=verbose)

    # update sample_p if in crop mode
    if crop:
      if self.opt["weights"]["rest_1v1"] > 0:
        self.loss_history[0] = self.aux["log"]["rest_1v1"]
      if self.opt["weights"]["rest_1vN"] > 0:
        self.loss_history[1] = self.aux["log"]["rest_1vN"]
      if self.opt["weights"]["rest_MvN"] > 0:
        self.loss_history[2] = self.aux["log"]["rest_MvN"]
      if self.opt["weights"]["rest_non"] > 0:
        self.loss_history[3] = self.aux["log"]["rest_non"]

      # whether all kind of possible rest loss have obtained
      flag_loss = [0 if iloss is None else 1 for iloss in self.loss_history]
      flag_loss = np.array(flag_loss)
      flag_valid = np.array(self.flag_valid)
      if (flag_loss * flag_valid).sum() == flag_valid.sum():
        loss_sample = [0 if iloss is None else iloss for iloss in self.loss_history]
        loss_sample = np.array(loss_sample) + pseudo_loss
        loss_sample *= flag_loss
        sample_p = loss_sample / (loss_sample.sum() + 1e-8)
        self.sample_p = sample_p.cumsum() * self._args["prob_rest"]

  def save_results(self, save_best=False, verbose=1):
    # save trajectory
    traj = {"log":self.aux["log"], "seq":np.asarray(self.aux["seq"]["pseudo"])}
    traj["xyz"] = np.asarray(self.aux["atom_positions"][:,1,:])
    traj["plddt"] = np.asarray(self.aux["plddt"])
    if "pae" in self.aux: traj["pae"] = np.asarray(self.aux["pae"])
    for k,v in traj.items(): self._traj[k].append(v)

    # save best result
    if save_best:
      metric = self.aux["log"][self._args["best_metric"]]
      if metric < self._best_metric:
        self._best_metric = self._best_loss = metric
        self._best_aux    = self._best_outs = self.aux
    
    if verbose and (self._k % verbose) == 0:
      # preferred order
      keys = ["models","recycles","hard","soft","temp","seqid","loss",
              "msa_ent","plddt","pae","helix","con","i_pae","i_con",
              "sc_fape","sc_rmsd","dgram_cce","fape","rmsd"]        
      print(dict_to_str(self.aux["log"], filt=self.opt["weights"],
                        print_str=f"{self._k}", keys=keys, ok="rmsd"))

  # ---------------------------------------------------------------------------------
  # example design functions
  # ---------------------------------------------------------------------------------
  def design(self, iters=100,
             soft=None, e_soft=None,
             temp=None, e_temp=None,
             hard=None, e_hard=None,
             opt=None, weights=None, dropout=None, crop=True,
             backprop=True, callback=None, save_best=False,
             verbose=1, grad_clip=False, save_every_n_step=1):

    # update options/settings (if defined)
    self.set_opt(opt, dropout=dropout)
    self.set_weights(weights)

    if soft is None: soft = self.opt["soft"]
    if temp is None: temp = self.opt["temp"]
    if hard is None: hard = self.opt["hard"]
    if e_soft is None: e_soft = soft
    if e_temp is None: e_temp = temp
    if e_hard is None: e_hard = hard

    loss_rest = []
    for i in range(iters):
      self.set_opt(soft=(soft+(e_soft-soft)*((i+1)/iters)),
                   hard=(hard+(e_hard-hard)*((i+1)/iters)),
                   temp=(e_temp+(temp-e_temp)*(1-(i+1)/iters)**2))
      
      # decay learning rate based on temperature
      lr_scale = (1 - self.opt["soft"]) + (self.opt["soft"] * self.opt["temp"])

      self.step(lr_scale=lr_scale, backprop=backprop, crop=crop,
                callback=callback, save_best=save_best, verbose=verbose, grad_clip=grad_clip, step_iter=i)

      #print(self.aux["inputs"])
      #print(self.aux["pdb"])
      if i % save_every_n_step == 0:
        self.gen_inputs.append(jax.tree_util.tree_map(np.array, self.aux["inputs"]))
        if self._args["crop"] is False:
          self.gen_outputs.append(jax.tree_util.tree_map(np.array, self.aux["pdb"]))
      
      if self._args["crop"] is False:
        iloss_rest = []
        if "rest_1v1" in self.aux["log"].keys():
          iloss_rest.append(self.aux["log"]["rest_1v1"])
        if "rest_1vN" in self.aux["log"].keys():
          iloss_rest.append(self.aux["log"]["rest_1vN"])
        if "rest_MvN" in self.aux["log"].keys():
          iloss_rest.append(self.aux["log"]["rest_MvN"])
        if "rest_non" in self.aux["log"].keys():
          iloss_rest.append(self.aux["log"]["rest_non"])
        loss_rest.append(iloss_rest)
        if iloss_rest and i > 10:
          loss_np = np.array(loss_rest)
          if (loss_np[-5:, :] < 0.05).sum() == 5*loss_np.shape[1]:
            break
