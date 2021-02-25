# coding=utf-8
"""
Anonymous author
"""
import os
import sys
import numpy as np
import math
from time import time
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from MaskGAF_rl import MaskedGraphAF

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit import DataStructs

import environment as env
#from environment import check_valency, convert_radical_electrons_to_hydrogens
from utils import save_one_mol, save_one_reward, update_optim_dict



def reward_target_molecule_similarity(mol, target, radius=2, nBits=2048,
                                      useChirality=True):
    """
    Reward for a target molecule similarity, based on tanimoto similarity
    between the ECFP fingerprints of the x molecule and target molecule
    :param mol: rdkit mol object
    :param target: rdkit mol object
    :return: float, [0.0, 1.0]
    """
    x = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=radius,
                                                        nBits=nBits,
                                                        useChirality=useChirality)
    target = rdMolDescriptors.GetMorganFingerprintAsBitVect(target,
                                                            radius=radius,
                                                        nBits=nBits,
                                                        useChirality=useChirality)
    return DataStructs.TanimotoSimilarity(x, target)




class GraphFlowModel(nn.Module):
    """
    Reminder:
        self.args: deq_coeff
                   deq_type

    Args:

    
    Returns:

    """
    def __init__(self, max_size, node_dim, bond_dim, edge_unroll, args):
        super(GraphFlowModel, self).__init__()
        self.max_size = max_size
        self.node_dim = node_dim
        self.bond_dim = bond_dim
        self.edge_unroll = edge_unroll
        self.args = args

        ###Flow hyper-paramters
        self.num_flow_layer = self.args.num_flow_layer
        self.nhid = self.args.nhid
        self.nout = self.args.nout

        self.node_masks = None
        if self.node_masks is None:
            self.node_masks, self.adj_masks, \
                self.link_prediction_index, self.flow_core_edge_masks = self.initialize_masks(max_node_unroll=self.max_size, max_edge_unroll=self.edge_unroll)

        self.latent_step = self.node_masks.size(0)  # (max_size) + (max_edge_unroll - 1) / 2 * max_edge_unroll + (max_size - max_edge_unroll) * max_edge_unroll
        self.latent_node_length = self.max_size * self.node_dim
        self.latent_edge_length = (self.latent_step - self.max_size) * self.bond_dim
        print('latent node length: %d' % self.latent_node_length)
        print('latent edge length: %d' % self.latent_edge_length)

        self.constant_pi = nn.Parameter(torch.Tensor([3.1415926535]), requires_grad=False)
        #learnable
        if self.args.learn_prior:
            self.prior_ln_var = nn.Parameter(torch.zeros([1])) # log(1^2) = 0
            nn.init.constant_(self.prior_ln_var, 0.)            
        else:
            self.prior_ln_var = nn.Parameter(torch.zeros([1]), requires_grad=False)

        self.dp = False
        num_gpus = torch.cuda.device_count()
        if num_gpus >= 1:
            self.dp = True
            print('using %d GPUs' % num_gpus)
        
        self.flow_core = MaskedGraphAF(self.node_masks, self.adj_masks, 
                                       self.link_prediction_index, 
                                       num_flow_layer = self.num_flow_layer,
                                       graph_size=self.max_size,
                                       num_node_type=self.node_dim,
                                       num_edge_type=self.bond_dim,
                                       args=self.args,
                                       nhid=self.nhid,
                                       nout=self.nout)

        self.flow_core_old = MaskedGraphAF(self.node_masks, self.adj_masks, 
                                       self.link_prediction_index, 
                                       num_flow_layer = self.num_flow_layer,
                                       graph_size=self.max_size,
                                       num_node_type=self.node_dim,
                                       num_edge_type=self.bond_dim,
                                       args=self.args,
                                       nhid=self.nhid,
                                       nout=self.nout)
        if self.dp:
            self.flow_core = nn.DataParallel(self.flow_core)
            self.flow_core_old = nn.DataParallel(self.flow_core_old)
        


    def forward(self, inp_node_features, inp_adj_features):
        """
        Args:
            inp_node_features: (B, N, 9)
            inp_adj_features: (B, 4, N, N)

        Returns:
            z: [(B, node_num*9), (B, edge_num*4)]
            logdet:  ([B], [B])        
        """
        #TODO: add dropout/normalize

        #inp_node_features_cont = inp_node_features #(B, N, 9) #! this is buggy. shallow copy
        inp_node_features_cont = inp_node_features.clone() #(B, N, 9)

        inp_adj_features_cont = inp_adj_features[:,:, self.flow_core_edge_masks].clone() #(B, 4, edge_num)
        inp_adj_features_cont = inp_adj_features_cont.permute(0, 2, 1).contiguous() #(B, edge_num, 4)


        if self.args.deq_type == 'random':
            #TODO: put the randomness on GPU.!
            inp_node_features_cont += self.args.deq_coeff * torch.rand(inp_node_features_cont.size()).cuda() #(B, N, 9)
            inp_adj_features_cont += self.args.deq_coeff * torch.rand(inp_adj_features_cont.size()).cuda() #(B, edge_num, 4)

        elif self.args.deq_type == 'variational':
            #TODO: add variational deq.
            raise ValueError('current unsupported method: %s' % self.args.deq_type)
        else:
            raise ValueError('unsupported dequantization type (%s)' % self.args.deq_type)


        z, logdet = self.flow_core(inp_node_features, inp_adj_features, 
                                   inp_node_features_cont, inp_adj_features_cont)
        
        if self.args.deq_type == 'random':
            return z, logdet, self.prior_ln_var

        elif self.args.deq_type == 'variational':
            #TODO: try variational dequantization
            return z, logdet, deq_logp, deq_logdet


    def reinforce_forward(self, temperature=0.75, mute=False, batch_size=32, max_size_rl=48, in_baseline=None, cur_iter=None):
        """
        Fintuning model using reinforce algorithm
        Args:
            temperature: generation temperature
            batch_size: batch_size for collecting data
            max_size_rl: maximal num of atoms allowed for generation

        Returns:

        """
        assert cur_iter is not None
        if cur_iter % self.args.update_iters == 0: # uodate the demenstration net every 4 iter.
            self.flow_core_old.load_state_dict(self.flow_core.state_dict())
            print('cur iter %d, updating old net' % cur_iter)
        #assert cur_baseline is not None
        num2bond = {0: Chem.rdchem.BondType.SINGLE, 1: Chem.rdchem.BondType.DOUBLE, 2: Chem.rdchem.BondType.TRIPLE}
        num2bond_symbol = {0: '=', 1: '==', 2: '==='}
        # [6, 7, 8, 9, 15, 16, 17, 35, 53, 0]
        num2atom = {0: 6, 1: 7, 2: 8, 3: 9, 4: 15, 5: 16, 6: 17, 7: 35, 8: 53}
        num2symbol = {0: 'C', 1: 'N', 2: 'O', 3: 'F', 4: 'P', 5: 'S', 6: 'Cl', 7: 'Br', 8: 'I'}

        prior_node_dist = torch.distributions.normal.Normal(torch.zeros([self.node_dim]).cuda(),
                                                            temperature * torch.ones([self.node_dim]).cuda())
        prior_edge_dist = torch.distributions.normal.Normal(torch.zeros([self.bond_dim]).cuda(),
                                                            temperature * torch.ones([self.bond_dim]).cuda())

        node_inputs = {}
        node_inputs['node_features'] = []
        node_inputs['adj_features'] = []
        node_inputs['node_features_cont'] = []
        node_inputs['rewards'] = []
        node_inputs['baseline_index'] = []

        adj_inputs = {}
        adj_inputs['node_features'] = []
        adj_inputs['adj_features'] = []
        adj_inputs['edge_features_cont'] = []
        adj_inputs['index'] = []
        adj_inputs['rewards'] = []
        adj_inputs['baseline_index'] = []

        #if in_baseline is not None:
        #    in_baseline = torch.Tensor(in_baseline).float().cuda()
        reward_baseline = torch.zeros([max_size_rl + 5, 2]).cuda()
        #reward_baseline[:, 0] += 1. #TODO: make sure all element in [:,0] is none zero

        max_action_size = 25 * \
                          (int(max_size_rl + (self.edge_unroll - 1) * self.edge_unroll / 2 + (max_size_rl-self.edge_unroll) * self.edge_unroll))

        batch_length = 0
        total_node_step = 0
        total_edge_step = 0

        per_mol_reward = []
        per_mol_property_score = []
        
        ### gather training data from generation
        self.eval() #! very important. Because we use batch normalization, training mode will result in unrealistic molecules
        
        with torch.no_grad():
            while total_node_step + total_edge_step < max_action_size and batch_length < batch_size:

                traj_node_inputs = {}
                traj_node_inputs['node_features'] = []
                traj_node_inputs['adj_features'] = []
                traj_node_inputs['node_features_cont'] = []
                traj_node_inputs['rewards'] = []
                traj_node_inputs['baseline_index'] = []
                traj_adj_inputs = {}
                traj_adj_inputs['node_features'] = []
                traj_adj_inputs['adj_features'] = []
                traj_adj_inputs['edge_features_cont'] = []
                traj_adj_inputs['index'] = []
                traj_adj_inputs['rewards'] = []
                traj_adj_inputs['baseline_index'] = []

                step_cnt = 1.0

                cur_node_features = torch.zeros([1, max_size_rl, self.node_dim]).cuda()
                cur_adj_features = torch.zeros([1, self.bond_dim, max_size_rl, max_size_rl]).cuda()

                rw_mol = Chem.RWMol()  # editable mol
                mol = None
                # mol_size = mol.GetNumAtoms()

                is_continue = True
                total_resample = 0
                each_node_resample = np.zeros([max_size_rl])

                step_num_data_edge = 0

                for i in range(max_size_rl):

                    step_num_data_edge = 0 # generating new node and its edges. Not sure if this will add into the final mol.

                    if not is_continue:
                        break
                    if i < self.edge_unroll:
                        edge_total = i  # edge to sample for current node
                        start = 0
                    else:
                        edge_total = self.edge_unroll
                        start = i - self.edge_unroll
                        
                    # first generate node
                    ## reverse flow
                    latent_node = prior_node_dist.sample().view(1, -1)  # (1, 9)
                    if self.dp:
                        latent_node = self.flow_core_old.module.reverse(cur_node_features, cur_adj_features,
                                                                    latent_node, mode=0).view(-1)  # (9, )
                    else:
                        latent_node = self.flow_core_old.reverse(cur_node_features, cur_adj_features,
                                                             latent_node, mode=0).view(-1)  # (9, )
                    ## node/adj postprocessing
                    # print(latent_node.shape) #(38, 9)
                    feature_id = torch.argmax(latent_node).item()
                    total_node_step += 1
                    node_feature_cont = torch.zeros([1, self.node_dim]).cuda()
                    node_feature_cont[0, feature_id] = 1.0


                    # update traj inputs for node_id
                    traj_node_inputs['node_features'].append(cur_node_features.clone())  # (1, max_size_rl, self.node_dim)
                    traj_node_inputs['adj_features'].append(cur_adj_features.clone())  # (1, self.bond_dim, max_size_rl, max_size_rl)
                    traj_node_inputs['node_features_cont'].append(node_feature_cont)  # (1, self.node_dim)
                    traj_node_inputs['rewards'].append(torch.full(size=(1,1), fill_value=step_cnt).cuda())  # (1, 1)
                    traj_node_inputs['baseline_index'].append(torch.full(size=(1,), fill_value=step_cnt).long().cuda())  # (1, 1)

                    #step_cnt += 1

                    # print(num2symbol[feature_id])
                    cur_node_features[0, i, feature_id] = 1.0
                    cur_adj_features[0, :, i, i] = 1.0
                    rw_mol.AddAtom(Chem.Atom(num2atom[feature_id]))

                    # then generate edges
                    if i == 0:
                        is_connect = True
                    else:
                        is_connect = False
                    # cur_mol_size = mol.GetNumAtoms
                    for j in range(edge_total):
                        valid = False
                        resample_edge = 0
                        invalid_bond_type_set = set()
                        while not valid:
                            if len(invalid_bond_type_set) < 3 and resample_edge <= 50:  # haven't sampled all possible bond type or is not stuck in the loop
                                latent_edge = prior_edge_dist.sample().view(1, -1)  # (1, 4)
                                if self.dp:
                                    latent_edge = self.flow_core_old.module.reverse(cur_node_features, cur_adj_features, latent_edge,
                                                                                mode=1, edge_index=torch.Tensor([[j + start, i]]).long().cuda()).view(-1)  # (4, )
                                else:
                                    latent_edge = self.flow_core_old.reverse(cur_node_features, cur_adj_features, latent_edge, mode=1,
                                                                edge_index=torch.Tensor([[j + start, i]]).long().cuda()).view(-1)  # (4, )
                                #if mol.GetNumAtoms() < self.args.min_atoms and j == (edge_total - 1): # molecule is small and current edge is last
                                #    latent_edge[-1] = -999999. #Do argmax in first 3 dimension
                                edge_discrete_id = torch.argmax(latent_edge).item()
                            else:
                                if not mute:
                                    print('have tried all possible bond type, use virtual bond.')
                                assert resample_edge > 50 or len(invalid_bond_type_set) == 3
                                edge_discrete_id = 3  # we have no choice but to choose not to add edge between (i, j+start)
                            total_edge_step += 1
                            edge_feature_cont = torch.zeros([1, self.bond_dim]).cuda()
                            edge_feature_cont[0, edge_discrete_id] = 1.0

                            # update traj inputs for edge_id
                            traj_adj_inputs['node_features'].append(cur_node_features.clone())  # 1, max_size_rl, self.node_dim
                            traj_adj_inputs['adj_features'].append(cur_adj_features.clone())  # 1, self.bond_dim, max_size_rl, max_size_rl
                            traj_adj_inputs['edge_features_cont'].append(edge_feature_cont)  # 1, self.bond_dim
                            traj_adj_inputs['index'].append(torch.Tensor([[j + start, i]]).long().cuda().view(1,-1)) # (1, 2)
                            step_num_data_edge += 1 # add one edge data, not sure if this should be added to the final train data

                            cur_adj_features[0, edge_discrete_id, i, j + start] = 1.0
                            cur_adj_features[0, edge_discrete_id, j + start, i] = 1.0
                            if edge_discrete_id == 3:  # virtual edge
                                valid = True # virtual edge is alway valid
                            else:  # single/double/triple bond
                                rw_mol.AddBond(i, j + start, num2bond[edge_discrete_id])
                                valid = env.check_valency(rw_mol)
                                if valid:
                                    is_connect = True
                                    # print(num2bond_symbol[edge_discrete_id])
                                else:  # backtrack
                                    rw_mol.RemoveBond(i, j + start)
                                    cur_adj_features[0, edge_discrete_id, i, j + start] = 0.0
                                    cur_adj_features[0, edge_discrete_id, j + start, i] = 0.0
                                    total_resample += 1.0
                                    each_node_resample[i] += 1.0
                                    resample_edge += 1

                                    invalid_bond_type_set.add(edge_discrete_id)

                            if valid:
                                traj_adj_inputs['rewards'].append(torch.full(size=(1, 1), fill_value=step_cnt).cuda())  # (1, 1)
                                traj_adj_inputs['baseline_index'].append(torch.full(size=(1,), fill_value=step_cnt).long().cuda())  # (1)

                                #step_cnt += 1
                            else:
                                if self.args.penalty:
                                    # todo: the step_reward can be tuned here
                                    traj_adj_inputs['rewards'].append(torch.full(size=(1, 1), fill_value=-1.).cuda())  # (1, 1) invalid edge penalty
                                    traj_adj_inputs['baseline_index'].append(torch.full(size=(1,), fill_value=step_cnt).long().cuda())  # (1,)
                                    #TODO: check baselien of invalid step, maybe we do not add baseline for invalid step
                                else:
                                    traj_adj_inputs['node_features'].pop(-1)
                                    traj_adj_inputs['adj_features'].pop(-1)
                                    traj_adj_inputs['edge_features_cont'].pop(-1)
                                    traj_adj_inputs['index'].pop(-1)
                                    step_num_data_edge -= 1 # if we do not penalize invalid edge, pop train data, decrease counter by 1                              

                                    

                    if is_connect:  # new generated node has at least one bond with previous node, do not stop generation, backup mol from rw_mol to mol
                        is_continue = True
                        mol = rw_mol.GetMol()

                    else:
                        is_continue = False
                    step_cnt += 1

                #TODO: check the last iter of generation
                #(Thinking)
                # The last node was not added. So after we generate the second to last
                # node and all its edges, the rest adjacent matrix and node features should all be zero
                # But current implementation append
                num_atoms = mol.GetNumAtoms()
                assert num_atoms <= max_size_rl

                #TODO: check if we should discard the small molecule
                #if num_atoms < self.args.min_atoms:
                #    print('#atoms of generated molecule less than %d, discarded!' % self.args.min_atoms)
                #    continue
                #else:
                #    batch_length += 1
                #    print('generating %d-th molecule done!' % batch_length)
                batch_length += 1
                print('generating %d-th molecule with %d atoms' % (batch_length, num_atoms))

                if num_atoms < max_size_rl:   
                    #! this implementation is buggy. we only mask the last node feature cont
                    #! But we ignore the non-zero node features in generating edges
                    #! this pattern will make model not to generated any edges between
                    #! the new-generated isolated node and exsiting subgraph.
                    #! this may be the biggest bug in Reinforce algorithm!!!!!
                    #! since the final iter/(step) has largest reward....!!!!!!!
                    #! work around1: add a counter and mask out all node feautres in generating edges of last iter.
                    #! work around2: do not append any data if the isolated node is not connected to subgraph.
                    # currently use work around2

                    # pop all the reinforce train-data add by at the generating the last isolated node and its edge
                    ## pop node
                    traj_node_inputs['node_features'].pop(-1)
                    traj_node_inputs['adj_features'].pop(-1)
                    traj_node_inputs['node_features_cont'].pop(-1)
                    traj_node_inputs['rewards'].pop(-1)
                    traj_node_inputs['baseline_index'].pop(-1)
                   
                    ## pop adj
                    for pop_cnt in range(step_num_data_edge):
                        traj_adj_inputs['node_features'].pop(-1)
                        traj_adj_inputs['adj_features'].pop(-1)
                        traj_adj_inputs['edge_features_cont'].pop(-1)
                        traj_adj_inputs['index'].pop(-1)
                        traj_adj_inputs['rewards'].pop(-1)
                        traj_adj_inputs['baseline_index'].pop(-1)

                    # generated molecule doesn't reach the maximal size
                    # last node's feature should be all zero
                    #traj_node_inputs['node_features_cont'][-1] = torch.zeros([1, self.node_dim]).cuda()              

                reward_valid = 2
                reward_property = 0
                reward_length = 0 
                #TODO: check if it is necessary to penal the small molecule, current we do not use it
                #if num_atoms < 10:
                #    reward_length -= 1.
                #elif num_atoms >= 20
                #    reward_length += 1
                #elif num_atoms >= 30:
                #    reward_length += 4
                #reward_length = 0  # current we do not use this reward
                flag_steric_strain_filter = True
                flag_zinc_molecule_filter = True

                assert mol is not None, 'mol is None...'
                final_valid = env.check_chemical_validity(mol)

                assert final_valid is True, 'warning: use valency check during generation but the final molecule is invalid!!!'
                if not final_valid:
                    reward_valid -= 5 # this is useless, because this case will not occur in our implementation
                else:
                    final_mol = env.convert_radical_electrons_to_hydrogens(mol)
                    s = Chem.MolToSmiles(final_mol, isomericSmiles=True)
                    print(s)
                    final_mol = Chem.MolFromSmiles(s)
                    # mol filters with negative rewards
                    if not env.steric_strain_filter(final_mol):  # passes 3D conversion, no excessive strain
                        reward_valid -= 1 #TODO: check the magnitude of this reward.
                        flag_steric_strain_filter = False
                    if not env.zinc_molecule_filter(final_mol):  # does not contain any problematic functional groups
                        reward_valid -= 1
                        flag_zinc_molecule_filter = False

                    # todo: add arg for property_type here
                    property_type = self.args.property
                    assert property_type in ['qed', 'plogp'], 'unsupported property optimization, choices are [qed, plogp]'

                    try:
                        if property_type == 'qed':
                            score = env.qed(final_mol) # value in [0, 1]

                            if self.args.reward_type == 'exp':
                                reward_property += (np.exp(score / self.args.exp_temperature) - self.args.exp_bias)

                            elif self.args.reward_type == 'linear':
                                reward_property += (score * self.args.qed_coeff)

                            if score > 0.940:
                                save_one_mol_path = os.path.join(self.args.save_path, 'good_mol_qed.txt')
                                save_one_mol(save_one_mol_path, s, cur_iter=cur_iter, score=score)

                        elif property_type == 'plogp':
                            score = env.penalized_logp(final_mol) # unbounded value

                            #TODO: design stable reward....
                            if self.args.reward_type == 'exp':
                                reward_property += (np.exp(score / self.args.exp_temperature) - self.args.exp_bias)  
                            elif self.args.reward_type == 'linear':
                                reward_property += (score * self.args.plogp_coeff)
                            #elif self.args.reward_type == 'exp_with_linear':
                            #    reward_property += (score * self.args.plogp_coeff + np.exp(score / self.args.exp_temperature))
                            #elif self.args.reward_type == 'neg_exp': # not working well
                            #    if score < 0:
                            #        tmp_score = -2
                            #    else:
                            #        tmp_score = np.exp(score / self.args.exp_temperature - 1.)
                            #    reward_property += (tmp_score) 
                            #score_tmp = score * self.args.plogp_coeff
                            #reward_property += (np.exp(score_tmp / self.args.exp_temperature) - 4.0)  
                            
                            if score > 4.0:
                                save_one_mol_path = os.path.join(self.args.save_path, 'good_mol_plogp.txt')
                                save_one_mol(save_one_mol_path, s, cur_iter=cur_iter, score=score)                                
                    except:
                        print('generated mol does not pass env.qed/plogp')
                        #reward_property -= 2.0 
                        #TODO: check what we should do if the model do not pass qed/plogp test
                        # workaround1: add a extra penalty reward
                        # workaround2: discard the molecule.

                reward_final_total = reward_valid + reward_property + reward_length
                #reward_final_total = reward_property
                per_mol_reward.append(reward_final_total)
                per_mol_property_score.append(reward_property)

                # todo: add arg for reward decay weight here
                reward_decay = self.args.reward_decay


                node_inputs['node_features'].append(torch.cat(traj_node_inputs['node_features'], dim=0)) #append tensor of shape (max_size_rl, max_size_rl, self.node_dim)
                node_inputs['adj_features'].append(torch.cat(traj_node_inputs['adj_features'], dim=0)) # append tensor of shape (max_size_rl, bond_dim, max_size_rl, max_size_rl)
                node_inputs['node_features_cont'].append(torch.cat(traj_node_inputs['node_features_cont'], dim=0)) # append tensor of shape (max_size_rl, 9)

                traj_node_inputs_baseline_index = torch.cat(traj_node_inputs['baseline_index'], dim=0) #(max_size_rl)
                traj_node_inputs_rewards = torch.cat(traj_node_inputs['rewards'], dim=0) # tensor of shape (max_size_rl, 1)
                traj_node_inputs_rewards[traj_node_inputs_rewards > 0] = \
                    reward_final_total * torch.pow(reward_decay, step_cnt - 1. - traj_node_inputs_rewards[traj_node_inputs_rewards > 0])
                node_inputs['rewards'].append(traj_node_inputs_rewards)  # append tensor of shape (max_size_rl, 1)                
                node_inputs['baseline_index'].append(traj_node_inputs_baseline_index)

                for ss in range(traj_node_inputs_rewards.size(0)):
                    reward_baseline[traj_node_inputs_baseline_index[ss]][0] += 1.0
                    reward_baseline[traj_node_inputs_baseline_index[ss]][1] += traj_node_inputs_rewards[ss][0]                
                

                adj_inputs['node_features'].append(torch.cat(traj_adj_inputs['node_features'], dim=0)) # (step, max_size_rl, self.node_dim)
                adj_inputs['adj_features'].append(torch.cat(traj_adj_inputs['adj_features'], dim=0)) # (step, bond_dim, max_size_rl, max_size_rl)
                adj_inputs['edge_features_cont'].append(torch.cat(traj_adj_inputs['edge_features_cont'], dim=0)) # (step, 4)
                adj_inputs['index'].append(torch.cat(traj_adj_inputs['index'], dim=0)) # (step, 2)

                traj_adj_inputs_baseline_index = torch.cat(traj_adj_inputs['baseline_index'], dim=0) #(step)                
                traj_adj_inputs_rewards = torch.cat(traj_adj_inputs['rewards'], dim=0)
                traj_adj_inputs_rewards[traj_adj_inputs_rewards > 0] = \
                    reward_final_total * torch.pow(reward_decay, step_cnt - 1. - traj_adj_inputs_rewards[traj_adj_inputs_rewards > 0])
                adj_inputs['rewards'].append(traj_adj_inputs_rewards)
                adj_inputs['baseline_index'].append(traj_adj_inputs_baseline_index)

                for ss in range(traj_adj_inputs_rewards.size(0)):
                    reward_baseline[traj_adj_inputs_baseline_index[ss]][0] += 1.0
                    reward_baseline[traj_adj_inputs_baseline_index[ss]][1] += traj_adj_inputs_rewards[ss][0]

        self.flow_core.train()
        #TODO: check whether to finetuning batch normalization
        for module in self.modules():
            if isinstance(module, torch.nn.modules.BatchNorm1d):
                module.eval()

        for i in range(reward_baseline.size(0)):
            if reward_baseline[i, 0] == 0:
                reward_baseline[i, 0] += 1.

        reward_baseline_per_step = reward_baseline[:, 1] / reward_baseline[:, 0] # (max_size_rl, )
        #TODO: check the baseline for invalid edge penalty step....
        # panelty step do not have property reward. So its magnitude may be quite different from others.

        if in_baseline is not None:
        #    #print(reward_baseline_per_step.size())
            assert in_baseline.size() == reward_baseline_per_step.size()
            reward_baseline_per_step = reward_baseline_per_step * (1. - self.args.moving_coeff) + in_baseline * self.args.moving_coeff
            print('calculating moving baseline per step')

        node_inputs_node_features = torch.cat(node_inputs['node_features'], dim=0) # (total_size, max_size_rl, 9)
        node_inputs_adj_features = torch.cat(node_inputs['adj_features'], dim=0) # (total_size, 4, max_size_rl, max_size_rl)
        node_inputs_node_features_cont = torch.cat(node_inputs['node_features_cont'], dim=0) # (total_size, 9)
        node_inputs_rewards = torch.cat(node_inputs['rewards'], dim=0).view(-1) # (total_size,)
        node_inputs_baseline_index = torch.cat(node_inputs['baseline_index'], dim=0).long() # (total_size,)
        node_inputs_baseline = torch.index_select(reward_baseline_per_step, dim=0, index=node_inputs_baseline_index) #(total_size, )

        adj_inputs_node_features = torch.cat(adj_inputs['node_features'], dim=0) # (total_size, max_size_rl, 9)
        adj_inputs_adj_features = torch.cat(adj_inputs['adj_features'], dim=0) # (total_size, 4, max_size_rl, max_size_rl)
        adj_inputs_edge_features_cont = torch.cat(adj_inputs['edge_features_cont'], dim=0) # (total_size, 4)
        adj_inputs_index = torch.cat(adj_inputs['index'], dim=0) # (total_size, 2)
        adj_inputs_rewards = torch.cat(adj_inputs['rewards'], dim=0).view(-1) # (total_size,)
        adj_inputs_baseline_index = torch.cat(adj_inputs['baseline_index'], dim=0).long() #(total_size,)
        adj_inputs_baseline = torch.index_select(reward_baseline_per_step, dim=0, index=adj_inputs_baseline_index) #(total_size, )

        if self.args.deq_type == 'random':
            #! here comes the randomness from torch cpu
            # we put the device in torch.rand so it is genereated directly on gpu
            node_inputs_node_features_cont += self.args.deq_coeff * torch.rand(
                node_inputs_node_features_cont.size(), 
                device='cuda:%d' % (node_inputs_node_features_cont.get_device()))  # (total_size, 9)
            adj_inputs_edge_features_cont += self.args.deq_coeff * torch.rand(
                adj_inputs_edge_features_cont.size(),
                device='cuda:%d' % (adj_inputs_edge_features_cont.get_device()))  # (total_size, 4)

        elif self.args.deq_type == 'variational':
            # TODO: add variational deq.
            raise ValueError('current unsupported method: %s' % self.args.deq_type)
        else:
            raise ValueError('unsupported dequantization type (%s)' % self.args.deq_type)

        if self.dp:
            node_function = self.flow_core.module.forward_rl_node
            edge_function = self.flow_core.module.forward_rl_edge

            node_function_old = self.flow_core_old.module.forward_rl_node
            edge_function_old = self.flow_core_old.module.forward_rl_edge
        else:
            node_function = self.flow_core.forward_rl_node
            edge_function = self.flow_core.forward_rl_edge
            
            node_function_old = self.flow_core_old.forward_rl_node
            edge_function_old = self.flow_core_old.forward_rl_edge

        if not self.args.split_batch:
            z_node, logdet_node = node_function(node_inputs_node_features, node_inputs_adj_features,
                                                node_inputs_node_features_cont)  # (total_step, 9), (total_step, )

            z_edge, logdet_edge = edge_function(adj_inputs_node_features, adj_inputs_adj_features,
                                                adj_inputs_edge_features_cont, adj_inputs_index) # (total_step, 4), (total_step, )
        else:
            mid_node = node_inputs_node_features.size(0) // 2
            z_node1, logdet_node1 = node_function(node_inputs_node_features[:mid_node], node_inputs_adj_features[:mid_node],
                                                node_inputs_node_features_cont[:mid_node])  # (total_step, 9), (total_step, )
            z_node2, logdet_node2 = node_function(node_inputs_node_features[mid_node:], node_inputs_adj_features[mid_node:],
                                                node_inputs_node_features_cont[mid_node:])  # (total_step, 9), (total_step, )
            z_node = torch.cat([z_node1, z_node2], dim=0)                                                                                                            
            logdet_node = torch.cat([logdet_node1, logdet_node2], dim=0)

            mid_edge = adj_inputs_node_features.size(0) // 2
            z_edge1, logdet_edge1 = edge_function(adj_inputs_node_features[:mid_edge], adj_inputs_adj_features[:mid_edge],
                                                adj_inputs_edge_features_cont[:mid_edge], adj_inputs_index[:mid_edge]) # (total_step, 4), (total_step, )
            z_edge2, logdet_edge2 = edge_function(adj_inputs_node_features[mid_edge:], adj_inputs_adj_features[mid_edge:],
                                                adj_inputs_edge_features_cont[mid_edge:], adj_inputs_index[mid_edge:]) # (total_step, 4), (total_step, )                                                            
            z_edge = torch.cat([z_edge1, z_edge2], dim=0)
            logdet_edge = torch.cat([logdet_edge1, logdet_edge2], dim=0)


        with torch.no_grad():
            if not self.args.split_batch:
                z_node_old, logdet_node_old = node_function_old(node_inputs_node_features, node_inputs_adj_features,
                                                    node_inputs_node_features_cont)  # (total_step, 9), (total_step, )

                z_edge_old, logdet_edge_old = edge_function_old(adj_inputs_node_features, adj_inputs_adj_features,
                                                    adj_inputs_edge_features_cont, adj_inputs_index) # (total_step, 4), (total_step, )
            else:
                mid_node = node_inputs_node_features.size(0) // 2
                z_node_old1, logdet_node_old1 = node_function_old(node_inputs_node_features[:mid_node], node_inputs_adj_features[:mid_node],
                                                    node_inputs_node_features_cont[:mid_node])  # (total_step, 9), (total_step, )
                z_node_old2, logdet_node_old2 = node_function_old(node_inputs_node_features[mid_node:], node_inputs_adj_features[mid_node:],
                                                    node_inputs_node_features_cont[mid_node:])  # (total_step, 9), (total_step, )
                z_node_old = torch.cat([z_node_old1, z_node_old2], dim=0)
                logdet_node_old  = torch.cat([logdet_node_old1, logdet_node_old2], dim=0)                                                                   

                mid_edge = adj_inputs_node_features.size(0) // 2
                z_edge_old1, logdet_edge_old1 = edge_function_old(adj_inputs_node_features[:mid_edge], adj_inputs_adj_features[:mid_edge],
                                                    adj_inputs_edge_features_cont[:mid_edge], adj_inputs_index[:mid_edge]) # (total_step, 4), (total_step, )
                z_edge_old2, logdet_edge_old2 = edge_function_old(adj_inputs_node_features[mid_edge:], adj_inputs_adj_features[mid_edge:],
                                                    adj_inputs_edge_features_cont[mid_edge:], adj_inputs_index[mid_edge:]) # (total_step, 4), (total_step, )                
                z_edge_old = torch.cat([z_edge_old1, z_edge_old2], dim=0)
                logdet_edge_old = torch.cat([logdet_edge_old1, logdet_edge_old2], dim=0)




        node_total_length = z_node.size(0) * float(self.node_dim)
        edge_total_length = z_edge.size(0) * float(self.bond_dim)

        #logdet_node = logdet_node - self.latent_node_length  # calculate probability of a region from probability density, minus constant has no effect on optimization
        #logdet_edge = logdet_edge - self.latent_edge_length  # calculate probability of a region from probability density, minus constant has no effect on optimization

        ll_node = -1 / 2 * (torch.log(2 * self.constant_pi) + self.prior_ln_var + torch.exp(-self.prior_ln_var) * (z_node ** 2))
        ll_node = ll_node.sum(-1)  # (B)
        ll_edge = -1 / 2 * (torch.log(2 * self.constant_pi) + self.prior_ln_var + torch.exp(-self.prior_ln_var) * (z_edge ** 2))
        ll_edge = ll_edge.sum(-1)  # (B)

        ll_node += logdet_node  # ([B])
        ll_edge += logdet_edge  # ([B])


        ll_node_old = -1 / 2 * (torch.log(2 * self.constant_pi) + self.prior_ln_var + torch.exp(-self.prior_ln_var) * (z_node_old ** 2))
        ll_node_old = ll_node_old.sum(-1)  # (B)
        ll_edge_old = -1 / 2 * (torch.log(2 * self.constant_pi) + self.prior_ln_var + torch.exp(-self.prior_ln_var) * (z_edge_old ** 2))
        ll_edge_old = ll_edge_old.sum(-1)  # (B)

        ll_node_old += logdet_node_old  # ([B])
        ll_edge_old += logdet_edge_old  # ([B])



        #lb = 1. - 0.2
        #ub = 1. + 0.2

        #ratio_node_no_clamp = torch.exp((ll_node - ll_node_old.detach()).clamp(max=50.))
        #ratio_edge_no_clamp = torch.exp((ll_edge - ll_edge_old.detach()).clamp(max=50.))
        ratio_node = torch.exp((ll_node - ll_node_old.detach()).clamp(max=10., min=-10.))
        ratio_edge = torch.exp((ll_edge - ll_edge_old.detach()).clamp(max=10., min=-10.))        

        #ratio_node = torch.clamp(ll_node - ll_node_old.detach(), math.log(lb), math.log(ub))
        #ratio_edge = torch.clamp(ll_edge - ll_edge_old.detach(), math.log(lb), math.log(ub))

        #ratio_node = torch.exp(ratio_node)
        #ratio_node = torch.exp(ll_node - ll_node_old.detach())
        if torch.isinf(ratio_node).any():
            raise RuntimeError('ratio node has inf entries')
        #ratio_edge = torch.exp(ratio_edge)
        #ratio_edge = torch.exp(ll_edge - ll_edge_old.detach())
        if torch.isinf(ratio_edge).any():
            raise RuntimeError('ratio edge has inf entries')
        advantage_node = (node_inputs_rewards - node_inputs_baseline)
        advantage_edge = (adj_inputs_rewards - adj_inputs_baseline)

        surr1_node = ratio_node * advantage_node
        surr2_node = torch.clamp(ratio_node, 1-0.2, 1+0.2) * advantage_node
        #surr2_node = ratio_node * advantage_node

        surr1_edge = ratio_edge * advantage_edge
        surr2_edge = torch.clamp(ratio_edge, 1-0.2, 1+0.2) * advantage_edge
        #surr2_edge = ratio_edge * advantage_edge


        if torch.isnan(surr1_node).any():
            raise RuntimeError('surr1 node has NaN entries')
        if torch.isnan(surr2_node).any():
            raise RuntimeError('surr2 node has NaN entries')
        if torch.isnan(surr1_edge).any():
            raise RuntimeError('surr1 edge has NaN entries')
        if torch.isnan(surr2_edge).any():
            raise RuntimeError('surr2 edge has NaN entries')                        
        #TODO: check baseline of penalty step(invalid edge)
        #TODO: check whether moving baseline is better than batch average.
        #ll_node = ll_node * (node_inputs_rewards - node_inputs_baseline)
        #ll_edge = ll_edge * (adj_inputs_rewards - adj_inputs_baseline)          

        if self.args.deq_type == 'random':
            if self.args.divide_loss:
                #print(ll_node.size())
                #print(ll_edge.size())
                #print(torch.min(surr1_node, surr2_node))
                #print(torch.min(surr1_edge, surr2_edge))
                #torch.save(torch.min(surr1_node, surr2_node), os.path.join(self.args.save_path, 'surr_node_%d' % cur_iter))
                #torch.save(torch.min(surr1_edge, surr2_edge), os.path.join(self.args.save_path, 'surr_edge_%d' % cur_iter))
                return -((torch.min(surr1_node, surr2_node).sum() + torch.min(surr1_edge, surr2_edge).sum()) / (node_total_length + edge_total_length) - 1.0), per_mol_reward, per_mol_property_score, reward_baseline_per_step
            else:
                # ! useless
                return -torch.sum(ll_node + ll_edge) / batch_length  # scalar


    def reinforce_constrained_optim_one_mol(self, x, adj, mol_size, modify_size, raw_smile, bfs_perm_origin, 
                                                    temperature=0.75, mute=False, batch_size=1, max_size_rl=38):

        num2bond = {0: Chem.rdchem.BondType.SINGLE, 1: Chem.rdchem.BondType.DOUBLE, 2: Chem.rdchem.BondType.TRIPLE}
        num2bond_symbol = {0: '=', 1: '==', 2: '==='}
        # [6, 7, 8, 9, 15, 16, 17, 35, 53, 0]
        num2atom = {0: 6, 1: 7, 2: 8, 3: 9, 4: 15, 5: 16, 6: 17, 7: 35, 8: 53}
        num2symbol = {0: 'C', 1: 'N', 2: 'O', 3: 'F', 4: 'P', 5: 'S', 6: 'Cl', 7: 'Br', 8: 'I'}

        prior_node_dist = torch.distributions.normal.Normal(torch.zeros([self.node_dim]).cuda(),
                                                            temperature * torch.ones([self.node_dim]).cuda())
        prior_edge_dist = torch.distributions.normal.Normal(torch.zeros([self.bond_dim]).cuda(),
                                                            temperature * torch.ones([self.bond_dim]).cuda())            
        self.eval()

        cur_mol_smiles = []
        cur_mol_imps = []
        cur_mol_sims = []

        with torch.no_grad():
            flag_start_from_original_graph = False
            flag_reconstruct_from_node_adj = True

            rand_num = np.random.rand()
            if rand_num <= 0.5:
                cur_modify_size = np.random.randint(low=0, high=modify_size)
            else:
                cur_modify_size = 0

            if cur_modify_size == 0:
                flag_start_from_original_graph = True
            keep_size = mol_size - cur_modify_size
            
            org_bfs_perm_origin = bfs_perm_origin
            org_node_features = x #(1, 38, 9)
            org_adj_features = adj # (1, 4, 38, 38)

            cur_node_features = x.clone() #(1, 38, 9)
            cur_adj_features = adj.clone() # (1, 4, 38, 38)                

            cur_node_features[:, keep_size:, :] = 0.
            cur_adj_features[:, :, keep_size:, :] = 0.
            cur_adj_features[:, :, :, keep_size:] = 0.

            min_action_node = 1

            rw_mol = Chem.RWMol()  # editable mol
            #rw_mol_org = Chem.RWMol() # editable mol to reconstruct original mol
            mol = None
            #org_mol = None
            # mol_size = mol.GetNumAtoms()

            # reconstruct mol from existing subgraph
            # rw_mol stop update when it reach keep_size
            # rw_mol_org reconstruct the original molecule
            for i in range(mol_size):
                node_id = torch.argmax(org_node_features[0, i]).item()  #(9, )
                if i < keep_size:
                    rw_mol.AddAtom(Chem.Atom(num2atom[node_id]))
                    #rw_mol_org.AddAtom(Chem.Atom(num2atom[node_id]))
                else:
                    pass
                    #rw_mol_org.AddAtom(Chem.Atom(num2atom[node_id]))

                if i < self.edge_unroll:
                    edge_total = i  # edge to sample for current node
                    start = 0
                else:
                    edge_total = self.edge_unroll
                    start = i - self.edge_unroll

                for j in range(edge_total):
                    edge_id = torch.argmax(org_adj_features[0, :, i, j+start]).item() #(4, )
                    if edge_id == 3: # no bond to add
                        continue
                    if i < keep_size:
                        rw_mol.AddBond(i, j + start, num2bond[edge_id])
                    else:
                        pass

            mol = rw_mol.GetMol()
            #org_mol = rw_mol_org.GetMol()                
            #s_org = Chem.MolToSmiles(org_mol, isomericSmiles=True)
            s_raw = raw_smile
            org_mol_true_raw = Chem.MolFromSmiles(s_raw)

            #calculate property for mol org
            org_mol_plogp = env.calculate_min_plogp(org_mol_true_raw)

            if env.check_chemical_validity(mol) is False or (cur_modify_size == 0 and np.random.rand() <= 0.5): # do not use subgraph, use original graph
                print('subgraph is invalid, use original graph as subgraph')
                rw_mol = Chem.RWMol(org_mol_true_raw)
                mol = rw_mol.GetMol()
                flag_reconstruct_from_node_adj = False # bfs is not reflected in current mol
                keep_size = mol_size
                cur_node_features = org_node_features.clone()
                cur_adj_features = org_adj_features.clone()
                cur_node_features[:, keep_size:, :] = 0.
                cur_adj_features[:, :, keep_size:, :] = 0.
                cur_adj_features[:, :, :, keep_size:] = 0.                                        
                flag_start_from_original_graph = True

            assert env.check_chemical_validity(org_mol_true_raw) is True, 's_raw is %s' % (s_raw)
            assert env.check_chemical_validity(mol) is True

            assert mol.GetNumAtoms() == keep_size
            assert org_mol_true_raw.GetNumAtoms() == mol_size


            cur_node_features = cur_node_features.cuda()
            cur_adj_features = cur_adj_features.cuda()

            is_continue = True
            total_resample = 0
            each_node_resample = np.zeros([max_size_rl])

            #step_num_data_edge = 0

            added_num = 0
            node_features_each_iter_backup = cur_node_features.clone() # backup of features, updated when newly added node is connected to previous subgraph
            adj_features_each_iter_backup = cur_adj_features.clone()
            for i in range(keep_size, max_size_rl):
                #current_i_continue = True
                if not is_continue:
                    break                    
                #while current_i_continue is True:
                if 1: # ignore the indent
                    #step_num_data_edge = 0 # generating new node and its edges. Not sure if this will add into the final mol.


                    if i < self.edge_unroll:
                        edge_total = i  # edge to sample for current node
                        start = 0
                    else:
                        edge_total = self.edge_unroll
                        start = i - self.edge_unroll
                    
                    # first generate node
                    ## reverse flow
                    latent_node = prior_node_dist.sample().view(1, -1)  # (1, 9)
                    if self.dp:
                        latent_node = self.flow_core_old.module.reverse(cur_node_features, cur_adj_features,
                                                                    latent_node, mode=0).view(-1)  # (9, )
                    else:
                        latent_node = self.flow_core_old.reverse(cur_node_features, cur_adj_features,
                                                                latent_node, mode=0).view(-1)  # (9, )
                    ## node/adj postprocessing
                    # print(latent_node.shape) #(38, 9)
                    feature_id = torch.argmax(latent_node).item()
                    #node_feature_cont = torch.zeros([1, self.node_dim]).cuda()
                    #node_feature_cont[0, feature_id] = 1.0


                    #step_cnt += 1

                    # print(num2symbol[feature_id])
                    cur_node_features[0, i, feature_id] = 1.0
                    cur_adj_features[0, :, i, i] = 1.0
                    rw_mol.AddAtom(Chem.Atom(num2atom[feature_id]))

                    # then generate edges
                    if i == 0:
                        is_connect = True
                    else:
                        is_connect = False
                    # cur_mol_size = mol.GetNumAtoms
                    for j in range(edge_total):
                        valid = False
                        resample_edge = 0
                        invalid_bond_type_set = set()
                        while not valid:
                            if len(invalid_bond_type_set) < 3 and resample_edge <= 50:  # haven't sampled all possible bond type or is not stuck in the loop
                                latent_edge = prior_edge_dist.sample().view(1, -1)  # (1, 4)
                                if self.dp:
                                    latent_edge = self.flow_core_old.module.reverse(cur_node_features, cur_adj_features, latent_edge,
                                                                                mode=1, edge_index=torch.Tensor([[j + start, i]]).long().cuda()).view(-1)  # (4, )
                                else:
                                    latent_edge = self.flow_core_old.reverse(cur_node_features, cur_adj_features, latent_edge, mode=1,
                                                                edge_index=torch.Tensor([[j + start, i]]).long().cuda()).view(-1)  # (4, )
                                #if mol.GetNumAtoms() < self.args.min_atoms and j == (edge_total - 1): # molecule is small and current edge is last
                                #    latent_edge[-1] = -999999. #Do argmax in first 3 dimension
                                edge_discrete_id = torch.argmax(latent_edge).item()
                            else:
                                if not mute:
                                    print('have tried all possible bond type, use virtual bond.')
                                assert resample_edge > 50 or len(invalid_bond_type_set) == 3
                                edge_discrete_id = 3  # we have no choice but to choose not to add edge between (i, j+start)
                            #edge_feature_cont = torch.zeros([1, self.bond_dim]).cuda()
                            #edge_feature_cont[0, edge_discrete_id] = 1.0

                         
                            #step_num_data_edge += 1 # add one edge data, not sure if this should be added to the final train data

                            cur_adj_features[0, edge_discrete_id, i, j + start] = 1.0
                            cur_adj_features[0, edge_discrete_id, j + start, i] = 1.0
                            if edge_discrete_id == 3:  # virtual edge
                                valid = True # virtual edge is alway valid
                            else:  # single/double/triple bond
                                if flag_reconstruct_from_node_adj:
                                    rw_mol.AddBond(i, j + start, num2bond[edge_discrete_id])
                                else:
                                    #current rw_mol is copy from original_mol, which does not reflect the bfs order
                                    #so we use bfs_perm_origin to map j+start to the true node_id in rw_mol
                                    rw_mol.AddBond(i, int(org_bfs_perm_origin[j+start].item()), num2bond[edge_discrete_id])
                                valid = env.check_valency(rw_mol)
                                if valid:
                                    is_connect = True
                                    # print(num2bond_symbol[edge_discrete_id])
                                else:  # backtrack
                                    if flag_reconstruct_from_node_adj:
                                        rw_mol.RemoveBond(i, j + start)
                                    else:
                                        rw_mol.RemoveBond(i, int(org_bfs_perm_origin[j+start].item()))

                                    cur_adj_features[0, edge_discrete_id, i, j + start] = 0.0
                                    cur_adj_features[0, edge_discrete_id, j + start, i] = 0.0
                                    total_resample += 1.0
                                    each_node_resample[i] += 1.0
                                    resample_edge += 1

                                    invalid_bond_type_set.add(edge_discrete_id)
                            '''
                            if valid:
                                traj_adj_inputs['rewards'].append(torch.full(size=(1, 1), fill_value=step_cnt).cuda())  # (1, 1)
                                traj_adj_inputs['baseline_index'].append(torch.full(size=(1,), fill_value=step_cnt).long().cuda())  # (1)

                                #step_cnt += 1
                            else:
                                if self.args.penalty:
                                    # todo: the step_reward can be tuned here
                                    traj_adj_inputs['rewards'].append(torch.full(size=(1, 1), fill_value=-1.).cuda())  # (1, 1) invalid edge penalty
                                    traj_adj_inputs['baseline_index'].append(torch.full(size=(1,), fill_value=step_cnt).long().cuda())  # (1,)
                                    #TODO: check baselien of invalid step, maybe we do not add baseline for invalid step
                                else:
                                    traj_adj_inputs['node_features'].pop(-1)
                                    traj_adj_inputs['adj_features'].pop(-1)
                                    traj_adj_inputs['edge_features_cont'].pop(-1)
                                    traj_adj_inputs['index'].pop(-1)
                                    step_num_data_edge -= 1 # if we do not penalize invalid edge, pop train data, decrease counter by 1                              
                            '''
                                

                    if is_connect:  # new generated node has at least one bond with previous node, do not stop generation, backup mol from rw_mol to mol
                        is_continue = True
                        mol = rw_mol.GetMol()

                        if env.check_chemical_validity(mol) is True:
                            current_smile = Chem.MolToSmiles(mol, isomericSmiles=True)
                            tmp_mol1 = Chem.MolFromSmiles(current_smile)
                            current_imp = env.calculate_min_plogp(tmp_mol1) - org_mol_plogp
                            current_sim = reward_target_molecule_similarity(tmp_mol1, org_mol_true_raw)
                            if current_imp > 0:
                                cur_mol_smiles.append(current_smile)
                                cur_mol_imps.append(current_imp)
                                cur_mol_sims.append(current_sim)

                            if flag_reconstruct_from_node_adj is False: #not reconstructed from adj, can convert.
                                mol_converted = env.convert_radical_electrons_to_hydrogens(mol)                                        
                                if env.check_chemical_validity(mol_converted) is True:
                                    current_smile2 = Chem.MolToSmiles(mol_converted, isomericSmiles=True)
                                    tmp_mol2 = Chem.MolFromSmiles(current_smile2)
                                    current_imp2 = env.calculate_min_plogp(tmp_mol2) - org_mol_plogp
                                    current_sim2 = reward_target_molecule_similarity(tmp_mol2, org_mol_true_raw)
                                    if current_imp2 > 0:
                                        cur_mol_smiles.append(current_smile2)
                                        cur_mol_imps.append(current_imp2)
                                        cur_mol_sims.append(current_sim2)
                                       
                        node_features_each_iter_backup = cur_node_features.clone() # update node backup since new node is valid
                        adj_features_each_iter_backup = cur_adj_features.clone() #
                        added_num += 1
                        #current_i_continue = False

                    else:
                        #! we may need to satisfy min action here
                        if added_num >= min_action_node:
                            # if we have already add 'min_action_node', ignore and let continue be false.
                            # the data added in last iter will be pop afterwards
                            is_continue = False
                        else:
                            # added_num is less than min_action_node
                            # first pop the appended train data
                            is_continue = False # first set as False

                            print('taking min actions, %d remained' % (min_action_node-added_num))
                            rw_mol = Chem.RWMol(mol) # important, recover the mol
                            cur_node_features = node_features_each_iter_backup.clone() # recover the backuped features
                            cur_adj_features = adj_features_each_iter_backup.clone()
                        
                            # get_candidate                                
                            #***********************#
                            candidate = []
                            local_num2bond =  {0: Chem.rdchem.BondType.SINGLE, 1: Chem.rdchem.BondType.DOUBLE, 2: Chem.rdchem.BondType.TRIPLE}
                            local_bond2num =  {Chem.rdchem.BondType.SINGLE:1 , Chem.rdchem.BondType.DOUBLE:2, Chem.rdchem.BondType.TRIPLE:3}                                
                            cur_mol_size = rw_mol.GetNumAtoms()
                            if cur_mol_size < self.edge_unroll: # 12
                                candidate_start = 0
                                candidate_end = cur_mol_size
                            else:
                                candidate_start = cur_mol_size - self.edge_unroll
                                candidate_end = cur_mol_size

                            candidate_node_features = cur_node_features.clone() # (1, 38, 9)
                            candidate_adj_features = cur_adj_features.clone() # (1, 4, 38, 38)
                            candidate_node_features = candidate_node_features[0, candidate_start:candidate_end] #(12, 9)
                            candidate_adj_features = candidate_adj_features[0, :, candidate_start:candidate_end, candidate_start:candidate_end] #(4, 12, 12)
                            assert candidate_node_features.size(0) == candidate_adj_features.size(1)
                            for cur_cand_index in range(candidate_node_features.size(0)):
                                cur_node_feat = candidate_node_features[cur_cand_index] #(9,)
                                cur_adj_list = candidate_adj_features[:, cur_cand_index] #(4, 12,)
                                assert cur_node_feat.sum() == 1.0
                                if cur_node_feat[0] != 1.0:
                                    continue
                                cur_valency = 0
                                for neighbor_id in range(cur_adj_list.size(1)):
                                    if neighbor_id == cur_cand_index: # ignore self loop
                                        continue
                                    neighbor_edge_type = torch.argmax(cur_adj_list[:, neighbor_id]).item()
                                    assert neighbor_edge_type <= 3
                                    if neighbor_edge_type < 3:
                                        cur_valency += (neighbor_edge_type + 1)
                                if cur_valency > 0 and cur_valency <= 3:
                                    candidate.append(cur_cand_index + candidate_start)
                            print('selecting candidate done')
                            #***********************#

                            #! ***********************#
                            if len(candidate) > 0:
                                try_take_min_action_time = 5
                                flag_success = False
                                stop_sig = False
                                cur_try = 0
                                cur_try2 = 0
                                last_id1 = -1
                                cache_demon = [-1]
                                while cur_try < try_take_min_action_time and not stop_sig and cur_try2 < 2 * try_take_min_action_time:
                                    cur_try2 += 1
                                    last_id1 = -1
                                    #if 1:
                                    try:
                            
                                        mol_demon_edit = Chem.RWMol(rw_mol)
                                        #keep_size = org_mol_true_raw.GetNumAtoms()
                                        cur_node_features_tmp = cur_node_features.clone()
                                        cur_adj_features_tmp = cur_adj_features.clone()
                                        #cur_node_features[:, keep_size:, :] = 0.
                                        #cur_adj_features[:, :, keep_size:, :] = 0.
                                        #cur_adj_features[:, :, :, keep_size:] = 0.
                                        #cur_node_features = cur_node_features.cuda()  
                                        #cur_adj_features = cur_adj_features.cuda()
                                
                                        try_time=5
                                        try_count=0
                                        while last_id1 in cache_demon and try_count < try_time: # get a starting point
                                            try_count += 1
                                            last_id1 = np.random.randint(len(candidate))
                                            last_id1 = int(candidate[last_id1])
                                        cache_demon.append(last_id1)
                                        #print(org_bfs_perm_origin)
                                        if flag_reconstruct_from_node_adj:
                                            last_id1_origin = last_id1
                                        else:
                                            last_id1_origin = int(org_bfs_perm_origin[last_id1].item())
                                        #TODO:check the logic of bfs is right
                                        atom = mol_demon_edit.GetAtomWithIdx(last_id1_origin)
                                        print('start testing atom')
                                        assert atom.GetAtomicNum() == 6
                                        print('pass carbon test')
                                        assert sum([local_bond2num[x.GetBondType()] for x in atom.GetBonds()]) <= 3
                                        print('pass testing atom')
                                        for act_id in range(1): # only add one node and one edge
                                    
                                            last_id2 = mol_demon_edit.AddAtom(Chem.Atom(6))
                                            print('act_id:%d, org_mol_size: %d, cur mol size: %d, last_id1: %d, last_id2:%d' % (act_id, mol.GetNumAtoms(), mol_demon_edit.GetNumAtoms(), last_id1, last_id2))

                                            assert  mol_demon_edit.GetNumAtoms() == (last_id2 + 1)

                                            cur_node_features_tmp[0, last_id2, 0] = 1.0
                                            cur_adj_features_tmp[0, :, last_id2, last_id2] = 1.0
                                            mol_demon_edit.AddBond(last_id1_origin, last_id2, local_num2bond[0])
                                                                
                                            cur_adj_features_tmp[0, 0, last_id1, last_id2] = 1.0
                                            cur_adj_features_tmp[0, 0, last_id2, last_id1] = 1.0
                                        assert env.check_chemical_validity(mol_demon_edit) is True
                                        print('pass chemical validity check...')
                                        stop_sig = True
                                        flag_success = True
                                        rw_mol = Chem.RWMol(mol_demon_edit)
                                        cur_node_features = cur_node_features_tmp.clone()
                                        cur_adj_features = cur_adj_features_tmp.clone()
                                       
                                        cur_try += 1
                                     
                                    #if 1:
                                    except:
                                        print('demonstration invalid occur, continue, raw is: %s' % s_raw)
                                        continue
                                if flag_success: # successfully take one min action.
                                    #flag_use_demon = True
                                    is_continue = True
                                    mol = rw_mol.GetMol()

                                    if env.check_chemical_validity(mol) is True:
                                        current_smile = Chem.MolToSmiles(mol, isomericSmiles=True)
                                        tmp_mol1 = Chem.MolFromSmiles(current_smile)
                                        current_imp = env.calculate_min_plogp(tmp_mol1) - org_mol_plogp
                                        current_sim = reward_target_molecule_similarity(tmp_mol1, org_mol_true_raw)
                                        if current_imp > 0:
                                            cur_mol_smiles.append(current_smile)
                                            cur_mol_imps.append(current_imp)
                                            cur_mol_sims.append(current_sim)

                                        if flag_reconstruct_from_node_adj is False: #not reconstructed from adj, can convert.
                                            mol_converted = env.convert_radical_electrons_to_hydrogens(mol)                                        
                                            if env.check_chemical_validity(mol_converted) is True:
                                                current_smile2 = Chem.MolToSmiles(mol_converted, isomericSmiles=True)
                                                tmp_mol2 = Chem.MolFromSmiles(current_smile2)
                                                current_imp2 = env.calculate_min_plogp(tmp_mol2) - org_mol_plogp
                                                current_sim2 = reward_target_molecule_similarity(tmp_mol2, org_mol_true_raw)
                                                if current_imp2 > 0:
                                                    cur_mol_smiles.append(current_smile2)
                                                    cur_mol_imps.append(current_imp2)
                                                    cur_mol_sims.append(current_sim2)                                      
                                    node_features_each_iter_backup = cur_node_features.clone() # update node backup since new node is valid
                                    adj_features_each_iter_backup = cur_adj_features.clone() #
                                    added_num += 1                                        

                                    print('successfully take one action')
                                else:
                                    print('failed in take one min action')
                                    continue
                            
                            else:
                                continue
                            #! ***********************#
        return cur_mol_smiles, cur_mol_imps, cur_mol_sims




    def reinforce_forward_constrained_optim(self, mol_xs, mol_adjs, mol_sizes, modify_size, raw_smiles, bfs_perm_origin, 
                                    temperature=0.75, mute=False, batch_size=32, max_size_rl=48, in_baseline=None, cur_iter=None):
        """
        Fintuning model using reinforce algorithm
        Args:
            existing_mol: molecule to be optimized. Practically, we provide 64 mols per call and the function may take less then 64 mols
            temperature: generation temperature
            batch_size: batch_size for collecting data
            max_size_rl: maximal num of atoms allowed for generation

        Returns:

        """
        optim_dict = {} # key smile, value: [[s0, imp, sim], [s2,, imp, sim], [s4, imp, sim], [s6, imp, sim]]
        batch_size = min(batch_size, mol_sizes.size(0)) # last batch of one iter may be less batch_size

        assert cur_iter is not None
        if cur_iter % self.args.update_iters == 0: # uodate the demenstration net every 4 iter.
            self.flow_core_old.load_state_dict(self.flow_core.state_dict())
            print('cur iter %d, updating old net' % cur_iter)
        #assert cur_baseline is not None
        num2bond = {0: Chem.rdchem.BondType.SINGLE, 1: Chem.rdchem.BondType.DOUBLE, 2: Chem.rdchem.BondType.TRIPLE}
        num2bond_symbol = {0: '=', 1: '==', 2: '==='}
        # [6, 7, 8, 9, 15, 16, 17, 35, 53, 0]
        num2atom = {0: 6, 1: 7, 2: 8, 3: 9, 4: 15, 5: 16, 6: 17, 7: 35, 8: 53}
        num2symbol = {0: 'C', 1: 'N', 2: 'O', 3: 'F', 4: 'P', 5: 'S', 6: 'Cl', 7: 'Br', 8: 'I'}

        prior_node_dist = torch.distributions.normal.Normal(torch.zeros([self.node_dim]).cuda(),
                                                            temperature * torch.ones([self.node_dim]).cuda())
        prior_edge_dist = torch.distributions.normal.Normal(torch.zeros([self.bond_dim]).cuda(),
                                                            temperature * torch.ones([self.bond_dim]).cuda())

        node_inputs = {}
        node_inputs['node_features'] = []
        node_inputs['adj_features'] = []
        node_inputs['node_features_cont'] = []
        node_inputs['rewards'] = []
        node_inputs['baseline_index'] = []

        adj_inputs = {}
        adj_inputs['node_features'] = []
        adj_inputs['adj_features'] = []
        adj_inputs['edge_features_cont'] = []
        adj_inputs['index'] = []
        adj_inputs['rewards'] = []
        adj_inputs['baseline_index'] = []

        #if in_baseline is not None:
        #    in_baseline = torch.Tensor(in_baseline).float().cuda()
        reward_baseline = torch.zeros([max_size_rl + 5, 2]).cuda()
        #reward_baseline[:, 0] += 1. #TODO: make sure all element in [:,0] is none zero

        max_action_size = 32 * \
                          (int(max_size_rl + (self.edge_unroll - 1) * self.edge_unroll / 2 + (max_size_rl-self.edge_unroll) * self.edge_unroll))

        batch_length = 0
        total_node_step = 0
        total_edge_step = 0

        per_mol_reward = []
        per_mol_property_score = []
        
        ### gather training data from generation
        self.eval() #! very important. Because we use batch normalization, training mode will result in unrealistic molecules
        
        with torch.no_grad():
            while total_node_step + total_edge_step < max_action_size and batch_length < batch_size:
                
                traj_node_inputs = {}
                traj_node_inputs['node_features'] = []
                traj_node_inputs['adj_features'] = []
                traj_node_inputs['node_features_cont'] = []
                traj_node_inputs['rewards'] = []
                traj_node_inputs['baseline_index'] = []
                traj_adj_inputs = {}
                traj_adj_inputs['node_features'] = []
                traj_adj_inputs['adj_features'] = []
                traj_adj_inputs['edge_features_cont'] = []
                traj_adj_inputs['index'] = []
                traj_adj_inputs['rewards'] = []
                traj_adj_inputs['baseline_index'] = []

                step_cnt = 1.0

                flag_start_from_original_graph = False
                flag_reconstruct_from_node_adj = True
                flag_use_demon = False

                rand_num = np.random.rand()
                if rand_num <= 0.5:
                    cur_modify_size = np.random.randint(low=0, high=modify_size)
                else:
                    cur_modify_size = 0
                #print(cur_modify_size)
                if cur_modify_size == 0:
                    flag_start_from_original_graph = True
                keep_size = mol_sizes[batch_length] - cur_modify_size
                
                org_bfs_perm_origin = bfs_perm_origin[batch_length]
                org_node_features = mol_xs[batch_length:batch_length+1] #(1, 38, 9)
                org_adj_features = mol_adjs[batch_length:batch_length+1] # (1, 4, 38, 38)

                cur_node_features = mol_xs[batch_length:batch_length+1].clone() #(1, 38, 9)
                cur_adj_features = mol_adjs[batch_length:batch_length+1].clone() # (1, 4, 38, 38)                

                cur_node_features[:, keep_size:, :] = 0.
                cur_adj_features[:, :, keep_size:, :] = 0.
                cur_adj_features[:, :, :, keep_size:] = 0.

                min_action_node = 1

                rw_mol = Chem.RWMol()  # editable mol
                #rw_mol_org = Chem.RWMol() # editable mol to reconstruct original mol
                mol = None
                #org_mol = None
                # mol_size = mol.GetNumAtoms()

                # reconstruct mol from existing subgraph
                # rw_mol stop update when it reach keep_size
                # rw_mol_org reconstruct the original molecule
                for i in range(mol_sizes[batch_length]):
                    node_id = torch.argmax(org_node_features[0, i]).item()  #(9, )
                    if i < keep_size:
                        rw_mol.AddAtom(Chem.Atom(num2atom[node_id]))
                        #rw_mol_org.AddAtom(Chem.Atom(num2atom[node_id]))
                    else:
                        pass
                        #rw_mol_org.AddAtom(Chem.Atom(num2atom[node_id]))


                    if i < self.edge_unroll:
                        edge_total = i  # edge to sample for current node
                        start = 0
                    else:
                        edge_total = self.edge_unroll
                        start = i - self.edge_unroll

                    for j in range(edge_total):
                        edge_id = torch.argmax(org_adj_features[0, :, i, j+start]).item() #(4, )
                        if edge_id == 3: # no bond to add
                            continue
                        if i < keep_size:
                            rw_mol.AddBond(i, j + start, num2bond[edge_id])
                            #rw_mol_org.AddBond(i, j + start, num2bond[edge_id])
                        else:
                            pass
                            #rw_mol_org.AddBond(i, j + start, num2bond[edge_id])

                mol = rw_mol.GetMol()
                #org_mol = rw_mol_org.GetMol()                
                #s_org = Chem.MolToSmiles(org_mol, isomericSmiles=True)
                s_raw = raw_smiles[batch_length]
                org_mol_true_raw = Chem.MolFromSmiles(s_raw)

                if env.check_chemical_validity(mol) is False: # do not use subgraph, use original graph
                    print('subgraph is invalid, use original graph as subgraph')
                    rw_mol = Chem.RWMol(org_mol_true_raw)
                    mol = rw_mol.GetMol()
                    flag_reconstruct_from_node_adj = False # bfs is not reflected in current mol
                    keep_size = mol_sizes[batch_length]
                    cur_node_features = org_node_features.clone()
                    cur_adj_features = org_adj_features.clone()
                    cur_node_features[:, keep_size:, :] = 0.
                    cur_adj_features[:, :, keep_size:, :] = 0.
                    cur_adj_features[:, :, :, keep_size:] = 0.                                        
                    flag_start_from_original_graph = True

                assert env.check_chemical_validity(org_mol_true_raw) is True, 's_raw is %s' % (s_raw)
                assert env.check_chemical_validity(mol) is True

                assert mol.GetNumAtoms() == keep_size
                assert org_mol_true_raw.GetNumAtoms() == mol_sizes[batch_length]




                cur_node_features = cur_node_features.cuda()
                cur_adj_features = cur_adj_features.cuda()

                is_continue = True
                total_resample = 0
                each_node_resample = np.zeros([max_size_rl])

                step_num_data_edge = 0

                added_num = 0
                node_features_each_iter_backup = cur_node_features.clone() # backup of features, updated when newly added node is connected to previous subgraph
                adj_features_each_iter_backup = cur_adj_features.clone()
                for i in range(keep_size, max_size_rl):
                    #current_i_continue = True
                    if not is_continue:
                        break                    
                    #while current_i_continue is True:
                    if 1: # ignore the indent
                        step_num_data_edge = 0 # generating new node and its edges. Not sure if this will add into the final mol.


                        if i < self.edge_unroll:
                            edge_total = i  # edge to sample for current node
                            start = 0
                        else:
                            edge_total = self.edge_unroll
                            start = i - self.edge_unroll
                        
                        # first generate node
                        ## reverse flow
                        latent_node = prior_node_dist.sample().view(1, -1)  # (1, 9)
                        if self.dp:
                            latent_node = self.flow_core_old.module.reverse(cur_node_features, cur_adj_features,
                                                                        latent_node, mode=0).view(-1)  # (9, )
                        else:
                            latent_node = self.flow_core_old.reverse(cur_node_features, cur_adj_features,
                                                                 latent_node, mode=0).view(-1)  # (9, )
                        ## node/adj postprocessing
                        # print(latent_node.shape) #(38, 9)
                        feature_id = torch.argmax(latent_node).item()
                        total_node_step += 1
                        node_feature_cont = torch.zeros([1, self.node_dim]).cuda()
                        node_feature_cont[0, feature_id] = 1.0


                        # update traj inputs for node_id
                        traj_node_inputs['node_features'].append(cur_node_features.clone())  # (1, max_size_rl, self.node_dim)
                        traj_node_inputs['adj_features'].append(cur_adj_features.clone())  # (1, self.bond_dim, max_size_rl, max_size_rl)
                        traj_node_inputs['node_features_cont'].append(node_feature_cont)  # (1, self.node_dim)
                        traj_node_inputs['rewards'].append(torch.full(size=(1,1), fill_value=step_cnt).cuda())  # (1, 1)
                        traj_node_inputs['baseline_index'].append(torch.full(size=(1,), fill_value=step_cnt).long().cuda())  # (1, 1)

                        #step_cnt += 1

                        # print(num2symbol[feature_id])
                        cur_node_features[0, i, feature_id] = 1.0
                        cur_adj_features[0, :, i, i] = 1.0
                        rw_mol.AddAtom(Chem.Atom(num2atom[feature_id]))

                        # then generate edges
                        if i == 0:
                            is_connect = True
                        else:
                            is_connect = False
                        # cur_mol_size = mol.GetNumAtoms
                        for j in range(edge_total):
                            valid = False
                            resample_edge = 0
                            invalid_bond_type_set = set()
                            while not valid:
                                if len(invalid_bond_type_set) < 3 and resample_edge <= 50:  # haven't sampled all possible bond type or is not stuck in the loop
                                    latent_edge = prior_edge_dist.sample().view(1, -1)  # (1, 4)
                                    if self.dp:
                                        latent_edge = self.flow_core_old.module.reverse(cur_node_features, cur_adj_features, latent_edge,
                                                                                    mode=1, edge_index=torch.Tensor([[j + start, i]]).long().cuda()).view(-1)  # (4, )
                                    else:
                                        latent_edge = self.flow_core_old.reverse(cur_node_features, cur_adj_features, latent_edge, mode=1,
                                                                    edge_index=torch.Tensor([[j + start, i]]).long().cuda()).view(-1)  # (4, )
                                    #if mol.GetNumAtoms() < self.args.min_atoms and j == (edge_total - 1): # molecule is small and current edge is last
                                    #    latent_edge[-1] = -999999. #Do argmax in first 3 dimension
                                    edge_discrete_id = torch.argmax(latent_edge).item()
                                else:
                                    if not mute:
                                        print('have tried all possible bond type, use virtual bond.')
                                    assert resample_edge > 50 or len(invalid_bond_type_set) == 3
                                    edge_discrete_id = 3  # we have no choice but to choose not to add edge between (i, j+start)
                                total_edge_step += 1
                                edge_feature_cont = torch.zeros([1, self.bond_dim]).cuda()
                                edge_feature_cont[0, edge_discrete_id] = 1.0

                                # update traj inputs for edge_id
                                traj_adj_inputs['node_features'].append(cur_node_features.clone())  # 1, max_size_rl, self.node_dim
                                traj_adj_inputs['adj_features'].append(cur_adj_features.clone())  # 1, self.bond_dim, max_size_rl, max_size_rl
                                traj_adj_inputs['edge_features_cont'].append(edge_feature_cont)  # 1, self.bond_dim
                                traj_adj_inputs['index'].append(torch.Tensor([[j + start, i]]).long().cuda().view(1,-1)) # (1, 2)
                                step_num_data_edge += 1 # add one edge data, not sure if this should be added to the final train data

                                cur_adj_features[0, edge_discrete_id, i, j + start] = 1.0
                                cur_adj_features[0, edge_discrete_id, j + start, i] = 1.0
                                if edge_discrete_id == 3:  # virtual edge
                                    valid = True # virtual edge is alway valid
                                else:  # single/double/triple bond
                                    if flag_reconstruct_from_node_adj:
                                        rw_mol.AddBond(i, j + start, num2bond[edge_discrete_id])
                                    else:
                                        #current rw_mol is copy from original_mol, which does not reflect the bfs order
                                        #so we use bfs_perm_origin to map j+start to the true node_id in rw_mol
                                        rw_mol.AddBond(i, int(org_bfs_perm_origin[j+start].item()), num2bond[edge_discrete_id])
                                    valid = env.check_valency(rw_mol)
                                    if valid:
                                        is_connect = True
                                        # print(num2bond_symbol[edge_discrete_id])
                                    else:  # backtrack
                                        if flag_reconstruct_from_node_adj:
                                            rw_mol.RemoveBond(i, j + start)
                                        else:
                                            rw_mol.RemoveBond(i, int(org_bfs_perm_origin[j+start].item()))

                                        cur_adj_features[0, edge_discrete_id, i, j + start] = 0.0
                                        cur_adj_features[0, edge_discrete_id, j + start, i] = 0.0
                                        total_resample += 1.0
                                        each_node_resample[i] += 1.0
                                        resample_edge += 1

                                        invalid_bond_type_set.add(edge_discrete_id)

                                if valid:
                                    traj_adj_inputs['rewards'].append(torch.full(size=(1, 1), fill_value=step_cnt).cuda())  # (1, 1)
                                    traj_adj_inputs['baseline_index'].append(torch.full(size=(1,), fill_value=step_cnt).long().cuda())  # (1)

                                    #step_cnt += 1
                                else:
                                    if self.args.penalty:
                                        # todo: the step_reward can be tuned here
                                        traj_adj_inputs['rewards'].append(torch.full(size=(1, 1), fill_value=-1.).cuda())  # (1, 1) invalid edge penalty
                                        traj_adj_inputs['baseline_index'].append(torch.full(size=(1,), fill_value=step_cnt).long().cuda())  # (1,)
                                        #TODO: check baselien of invalid step, maybe we do not add baseline for invalid step
                                    else:
                                        traj_adj_inputs['node_features'].pop(-1)
                                        traj_adj_inputs['adj_features'].pop(-1)
                                        traj_adj_inputs['edge_features_cont'].pop(-1)
                                        traj_adj_inputs['index'].pop(-1)
                                        step_num_data_edge -= 1 # if we do not penalize invalid edge, pop train data, decrease counter by 1                              

                                    

                        if is_connect:  # new generated node has at least one bond with previous node, do not stop generation, backup mol from rw_mol to mol
                            is_continue = True
                            mol = rw_mol.GetMol()
                            node_features_each_iter_backup = cur_node_features.clone() # update node backup since new node is valid
                            adj_features_each_iter_backup = cur_adj_features.clone() #
                            added_num += 1
                            #current_i_continue = False

                        else:
                            #! we may need to satisfy min action here
                            if added_num >= min_action_node:
                                # if we have already add 'min_action_node', ignore and let continue be false.
                                # the data added in last iter will be pop afterwards
                                is_continue = False
                            else:
                                # added_num is less than min_action_node
                                # first pop the appended train data
                                is_continue = False # first set as False

                                print('taking min actions, %d remained' % (min_action_node-added_num))
                                rw_mol = Chem.RWMol(mol) # important, recover the mol
                                cur_node_features = node_features_each_iter_backup.clone() # recover the backuped features
                                cur_adj_features = adj_features_each_iter_backup.clone()
                                traj_node_inputs['node_features'].pop(-1)
                                traj_node_inputs['adj_features'].pop(-1)
                                traj_node_inputs['node_features_cont'].pop(-1)
                                traj_node_inputs['rewards'].pop(-1)
                                traj_node_inputs['baseline_index'].pop(-1)
                   
                                ## pop adj
                                for pop_cnt in range(step_num_data_edge):
                                    traj_adj_inputs['node_features'].pop(-1)
                                    traj_adj_inputs['adj_features'].pop(-1)
                                    traj_adj_inputs['edge_features_cont'].pop(-1)
                                    traj_adj_inputs['index'].pop(-1)
                                    traj_adj_inputs['rewards'].pop(-1)
                                    traj_adj_inputs['baseline_index'].pop(-1)
                                # get_candidate                                
                                #***********************#
                                candidate = []
                                local_num2bond =  {0: Chem.rdchem.BondType.SINGLE, 1: Chem.rdchem.BondType.DOUBLE, 2: Chem.rdchem.BondType.TRIPLE}
                                local_bond2num =  {Chem.rdchem.BondType.SINGLE:1 , Chem.rdchem.BondType.DOUBLE:2, Chem.rdchem.BondType.TRIPLE:3}                                
                                cur_mol_size = rw_mol.GetNumAtoms()
                                if cur_mol_size < self.edge_unroll: # 12
                                    candidate_start = 0
                                    candidate_end = cur_mol_size
                                else:
                                    candidate_start = cur_mol_size - self.edge_unroll
                                    candidate_end = cur_mol_size

                                candidate_node_features = cur_node_features.clone() # (1, 38, 9)
                                candidate_adj_features = cur_adj_features.clone() # (1, 4, 38, 38)
                                candidate_node_features = candidate_node_features[0, candidate_start:candidate_end] #(12, 9)
                                candidate_adj_features = candidate_adj_features[0, :, candidate_start:candidate_end, candidate_start:candidate_end] #(4, 12, 12)
                                assert candidate_node_features.size(0) == candidate_adj_features.size(1)
                                for cur_cand_index in range(candidate_node_features.size(0)):
                                    cur_node_feat = candidate_node_features[cur_cand_index] #(9,)
                                    cur_adj_list = candidate_adj_features[:, cur_cand_index] #(4, 12,)
                                    assert cur_node_feat.sum() == 1.0
                                    if cur_node_feat[0] != 1.0:
                                        continue
                                    cur_valency = 0
                                    for neighbor_id in range(cur_adj_list.size(1)):
                                        if neighbor_id == cur_cand_index: # ignore self loop
                                            continue
                                        neighbor_edge_type = torch.argmax(cur_adj_list[:, neighbor_id]).item()
                                        assert neighbor_edge_type <= 3
                                        if neighbor_edge_type < 3:
                                            cur_valency += (neighbor_edge_type + 1)
                                    if cur_valency > 0 and cur_valency <= 3:
                                        candidate.append(cur_cand_index + candidate_start)
                                print('selecting candidate done')
                                #***********************#

                                #! ***********************#
                                if len(candidate) > 0:
                                    try_take_min_action_time = 5
                                    flag_success = False
                                    stop_sig = False
                                    cur_try = 0
                                    cur_try2 = 0
                                    last_id1 = -1
                                    cache_demon = [-1]
                                    while cur_try < try_take_min_action_time and not stop_sig and cur_try2 < 2 * try_take_min_action_time:
                                        cur_try2 += 1
                                        last_id1 = -1
                                        #if 1:
                                        try:
                                            traj_node_inputs_tmp = {}
                                            traj_node_inputs_tmp['node_features'] = []
                                            traj_node_inputs_tmp['adj_features'] = []
                                            traj_node_inputs_tmp['node_features_cont'] = []
                                            traj_node_inputs_tmp['rewards'] = []
                                            traj_node_inputs_tmp['baseline_index'] = []
                                            traj_adj_inputs_tmp = {}
                                            traj_adj_inputs_tmp['node_features'] = []
                                            traj_adj_inputs_tmp['adj_features'] = []
                                            traj_adj_inputs_tmp['edge_features_cont'] = []
                                            traj_adj_inputs_tmp['index'] = []
                                            traj_adj_inputs_tmp['rewards'] = []
                                            traj_adj_inputs_tmp['baseline_index'] = []
                                            step_cnt_tmp = step_cnt # still use the current step cnt

                                            mol_demon_edit = Chem.RWMol(rw_mol)
                                            #keep_size = org_mol_true_raw.GetNumAtoms()
                                            cur_node_features_tmp = cur_node_features.clone()
                                            cur_adj_features_tmp = cur_adj_features.clone()
                                            #cur_node_features[:, keep_size:, :] = 0.
                                            #cur_adj_features[:, :, keep_size:, :] = 0.
                                            #cur_adj_features[:, :, :, keep_size:] = 0.
                                            #cur_node_features = cur_node_features.cuda()  
                                            #cur_adj_features = cur_adj_features.cuda()
                                    
                                            try_time=5
                                            try_count=0
                                            while last_id1 in cache_demon and try_count < try_time: # get a starting point
                                                try_count += 1
                                                last_id1 = np.random.randint(len(candidate))
                                                last_id1 = int(candidate[last_id1])
                                            cache_demon.append(last_id1)
                                            #print(org_bfs_perm_origin)
                                            if flag_reconstruct_from_node_adj:
                                                last_id1_origin = last_id1
                                            else:
                                                last_id1_origin = int(org_bfs_perm_origin[last_id1].item())
                                            #TODO:check the logic of bfs is right
                                            atom = mol_demon_edit.GetAtomWithIdx(last_id1_origin)
                                            print('start testing atom')
                                            assert atom.GetAtomicNum() == 6
                                            print('pass carbon test')
                                            assert sum([local_bond2num[x.GetBondType()] for x in atom.GetBonds()]) <= 3
                                            print('pass testing atom')
                                            for act_id in range(1): # only add one node and one edge
                                        
                                                last_id2 = mol_demon_edit.AddAtom(Chem.Atom(6))
                                                print('act_id:%d, org_mol_size: %d, cur mol size: %d, last_id1: %d, last_id2:%d' % (act_id, mol.GetNumAtoms(), mol_demon_edit.GetNumAtoms(), last_id1, last_id2))
                                                node_feature_cont = torch.zeros([1, self.node_dim]).cuda()
                                                node_feature_cont[0, 0] = 1.0   # carbon                                     
                                                traj_node_inputs_tmp['node_features'].append(cur_node_features_tmp.clone())  # (1, max_size_rl, self.node_dim)
                                                traj_node_inputs_tmp['adj_features'].append(cur_adj_features_tmp.clone())  # (1, self.bond_dim, max_size_rl, max_size_rl)
                                                traj_node_inputs_tmp['node_features_cont'].append(node_feature_cont)  # (1, self.node_dim)
                                                traj_node_inputs_tmp['rewards'].append(torch.full(size=(1,1), fill_value=step_cnt_tmp).cuda())  # (1, 1)
                                                traj_node_inputs_tmp['baseline_index'].append(torch.full(size=(1,), fill_value=step_cnt_tmp).long().cuda())  # (1, 1)                                        

                                                assert  mol_demon_edit.GetNumAtoms() == (last_id2 + 1)

                                                cur_node_features_tmp[0, last_id2, 0] = 1.0
                                                cur_adj_features_tmp[0, :, last_id2, last_id2] = 1.0
                                                #if act_id == 0:
                                                mol_demon_edit.AddBond(last_id1_origin, last_id2, local_num2bond[0])
                                                #else:
                                                #mol_demon_edit.AddBond(last_id1, last_id2, local_num2bond[0])
                                                edge_feature_cont = torch.zeros([1, self.bond_dim]).cuda()
                                                edge_feature_cont[0, 0] = 1.0  # single bond                                      
                                                traj_adj_inputs_tmp['node_features'].append(cur_node_features_tmp.clone())  # 1, max_size_rl, self.node_dim
                                                traj_adj_inputs_tmp['adj_features'].append(cur_adj_features_tmp.clone())  # 1, self.bond_dim, max_size_rl, max_size_rl
                                                traj_adj_inputs_tmp['edge_features_cont'].append(edge_feature_cont)  # 1, self.bond_dim
                                                traj_adj_inputs_tmp['index'].append(torch.Tensor([[last_id1, last_id2]]).long().cuda().view(1,-1)) # (1, 2)                                        
                                                traj_adj_inputs_tmp['rewards'].append(torch.full(size=(1, 1), fill_value=step_cnt_tmp).cuda())  # (1, 1)
                                                traj_adj_inputs_tmp['baseline_index'].append(torch.full(size=(1,), fill_value=step_cnt_tmp).long().cuda())  # (1)                                        
                                                cur_adj_features_tmp[0, 0, last_id1, last_id2] = 1.0
                                                cur_adj_features_tmp[0, 0, last_id2, last_id1] = 1.0
                                            assert env.check_chemical_validity(mol_demon_edit) is True
                                            print('pass chemical validity check...')
                                            stop_sig = True
                                            flag_success = True
                                            rw_mol = Chem.RWMol(mol_demon_edit)
                                            cur_node_features = cur_node_features_tmp.clone()
                                            cur_adj_features = cur_adj_features_tmp.clone()
                                            traj_node_inputs['node_features'].extend(traj_node_inputs_tmp['node_features'])
                                            traj_node_inputs['adj_features'].extend(traj_node_inputs_tmp['adj_features'])
                                            traj_node_inputs['node_features_cont'].extend(traj_node_inputs_tmp['node_features_cont'])
                                            traj_node_inputs['rewards'].extend(traj_node_inputs_tmp['rewards'])
                                            traj_node_inputs['baseline_index'].extend(traj_node_inputs_tmp['baseline_index'])

                                            traj_adj_inputs['node_features'].extend(traj_adj_inputs_tmp['node_features'])
                                            traj_adj_inputs['adj_features'].extend(traj_adj_inputs_tmp['adj_features'])
                                            traj_adj_inputs['edge_features_cont'].extend(traj_adj_inputs_tmp['edge_features_cont'])
                                            traj_adj_inputs['index'].extend(traj_adj_inputs_tmp['index'])
                                            traj_adj_inputs['rewards'].extend(traj_adj_inputs_tmp['rewards'])
                                            traj_adj_inputs['baseline_index'].extend(traj_adj_inputs_tmp['baseline_index'])
                                            cur_try += 1
                                            #print('act finish')
                                            #mol_demonstrated = mol_demon_edit.GetMol()
                                            #s_mol_demonstrated = Chem.MolToSmiles(mol_demonstrated, isomericSmiles=True)
                                            #mol = Chem.MolFromSmiles(s_mol_demonstrated)
                                            #print('mol get finish')
                                            #imp_demonstrated = env.calculate_min_plogp(mol) - env.calculate_min_plogp(org_mol_true_raw)
                                            #print('cal imp finish')
                                            #sim_demonstrated = reward_target_molecule_similarity(mol, org_mol_true_raw)
                                            #print('cal sim finish')
                                            #if imp_demonstrated > 0 and sim_demonstrated >= 0.6:
                                            #    stop_sig = True
                                            #    flag_success = True
                                            #else:
                                            #    stop_sig = False
                                            #cur_try += 1
                                        #if 1:
                                        except:
                                            print('demonstration invalid occur, continue, raw is: %s' % s_raw)
                                            continue
                                    if flag_success: # successfully take one min action.
                                        #flag_use_demon = True
                                        is_continue = True
                                        mol = rw_mol.GetMol()
                                        node_features_each_iter_backup = cur_node_features.clone() # update node backup since new node is valid
                                        adj_features_each_iter_backup = cur_adj_features.clone() #
                                        added_num += 1                                        

                                        print('successfully take one action')
                                    else:
                                        print('failed in take one min action')
                                        step_cnt += 1 # we should add one here since the loop is continued.
                                        continue
                                
                                else:
                                    step_cnt += 1
                                    continue




                                #! ***********************#




                    step_cnt += 1

                batch_length += 1

                #TODO: check the last iter of generation
                #(Thinking)
                # The last node was not added. So after we generate the second to last
                # node and all its edges, the rest adjacent matrix and node features should all be zero
                # But current implementation append
                num_atoms = mol.GetNumAtoms()
                assert num_atoms <= max_size_rl

                #TODO: check if we should discard the small molecule
                #if num_atoms < self.args.min_atoms:
                #    print('#atoms of generated molecule less than %d, discarded!' % self.args.min_atoms)
                #    continue
                #else:
                #    batch_length += 1
                #    print('generating %d-th molecule done!' % batch_length)
                print('generating %d-th molecule with %d atoms' % (batch_length, num_atoms))
                print('keep size is %d, newly add %d atoms, original size %d' % (keep_size, added_num, org_mol_true_raw.GetNumAtoms()))

                if num_atoms < max_size_rl:   
                    #! this implementation is buggy. we only mask the last node feature cont
                    #! But we ignore the non-zero node features in generating edges
                    #! this pattern will make model not to generated any edges between
                    #! the new-generated isolated node and exsiting subgraph.
                    #! this may be the biggest bug in Reinforce algorithm!!!!!
                    #! since the final iter/(step) has largest reward....!!!!!!!
                    #! work around1: add a counter and mask out all node feautres in generating edges of last iter.
                    #! work around2: do not append any data if the isolated node is not connected to subgraph.
                    # currently use work around2

                    # pop all the reinforce train-data add by at the generating the last isolated node and its edge
                    ## pop node
                    try:
                        traj_node_inputs['node_features'].pop(-1)
                        traj_node_inputs['adj_features'].pop(-1)
                        traj_node_inputs['node_features_cont'].pop(-1)
                        traj_node_inputs['rewards'].pop(-1)
                        traj_node_inputs['baseline_index'].pop(-1)
                   
                        ## pop adj
                        for pop_cnt in range(step_num_data_edge):
                            traj_adj_inputs['node_features'].pop(-1)
                            traj_adj_inputs['adj_features'].pop(-1)
                            traj_adj_inputs['edge_features_cont'].pop(-1)
                            traj_adj_inputs['index'].pop(-1)
                            traj_adj_inputs['rewards'].pop(-1)
                            traj_adj_inputs['baseline_index'].pop(-1)
                    except:
                        print('pop from empty list, take min action fail.')

                    # generated molecule doesn't reach the maximal size
                    # last node's feature should be all zero
                    #traj_node_inputs['node_features_cont'][-1] = torch.zeros([1, self.node_dim]).cuda()              

                if added_num == 0:
                    if flag_start_from_original_graph is False:
                        continue
                    else: # start from original graph, but policy stop add atom immediately, give guide
                        print('try to satisfy min action!')
                        local_num2bond =  {0: Chem.rdchem.BondType.SINGLE, 1: Chem.rdchem.BondType.DOUBLE, 2: Chem.rdchem.BondType.TRIPLE}
                        local_bond2num =  {Chem.rdchem.BondType.SINGLE:1 , Chem.rdchem.BondType.DOUBLE:2, Chem.rdchem.BondType.TRIPLE:3}
                        candidate = []

                        # get all candidate
                       

                        # method2: previous 12 nodes
                        #***********************#
                        org_mol_size = org_mol_true_raw.GetNumAtoms()
                        if org_mol_size < self.edge_unroll: # 12
                            candidate_start = 0
                            candidate_end = org_mol_size
                        else:
                            candidate_start = org_mol_size - self.edge_unroll
                            candidate_end = org_mol_size

                        candidate_node_features = org_node_features.clone() # (1, 38, 9)
                        candidate_adj_features = org_adj_features.clone() # (1, 4, 38, 38)
                        candidate_node_features = candidate_node_features[0, candidate_start:candidate_end] #(12, 9)
                        candidate_adj_features = candidate_adj_features[0, :, candidate_start:candidate_end, candidate_start:candidate_end] #(4, 12, 12)
                        assert candidate_node_features.size(0) == candidate_adj_features.size(1)
                        for cur_cand_index in range(candidate_node_features.size(0)):
                            cur_node_feat = candidate_node_features[cur_cand_index] #(9,)
                            cur_adj_list = candidate_adj_features[:, cur_cand_index] #(4, 12,)
                            assert cur_node_feat.sum() == 1.0
                            if cur_node_feat[0] != 1.0:
                                continue
                            cur_valency = 0
                            for neighbor_id in range(cur_adj_list.size(1)):
                                if neighbor_id == cur_cand_index: # ignore self loop
                                    continue
                                neighbor_edge_type = torch.argmax(cur_adj_list[:, neighbor_id]).item()
                                assert neighbor_edge_type <= 3
                                if neighbor_edge_type < 3:
                                    cur_valency += (neighbor_edge_type + 1)
                            if cur_valency > 0 and cur_valency <= 3:
                                candidate.append(cur_cand_index + candidate_start)
                        print('selecting candidate done')



                        #***********************#


                        if len(candidate) > 0:
                            demonstration_time = 10
                            flag_success = False
                            stop_sig = False
                            cur_try = 0
                            cur_try2 = 0
                            last_id1 = -1
                            cache_demon = [-1]
                            while cur_try < demonstration_time and not stop_sig and cur_try2 < 2 * demonstration_time:
                                cur_try2 += 1
                                last_id1 = -1
                                #if 1:
                                try:
                                    traj_node_inputs = {}
                                    traj_node_inputs['node_features'] = []
                                    traj_node_inputs['adj_features'] = []
                                    traj_node_inputs['node_features_cont'] = []
                                    traj_node_inputs['rewards'] = []
                                    traj_node_inputs['baseline_index'] = []
                                    traj_adj_inputs = {}
                                    traj_adj_inputs['node_features'] = []
                                    traj_adj_inputs['adj_features'] = []
                                    traj_adj_inputs['edge_features_cont'] = []
                                    traj_adj_inputs['index'] = []
                                    traj_adj_inputs['rewards'] = []
                                    traj_adj_inputs['baseline_index'] = []
                                    step_cnt = 1.0

                                    mol_demon_edit = Chem.RWMol(org_mol_true_raw)
                                    keep_size = org_mol_true_raw.GetNumAtoms()
                                    cur_node_features = org_node_features.clone()
                                    cur_adj_features = org_adj_features.clone()
                                    cur_node_features[:, keep_size:, :] = 0.
                                    cur_adj_features[:, :, keep_size:, :] = 0.
                                    cur_adj_features[:, :, :, keep_size:] = 0.
                                    cur_node_features = cur_node_features.cuda()  
                                    cur_adj_features = cur_adj_features.cuda()
                                    
                                    try_time=5
                                    try_count=0
                                    while last_id1 in cache_demon and try_count < try_time: # get a starting point
                                        try_count += 1
                                        last_id1 = np.random.randint(len(candidate))
                                        last_id1 = int(candidate[last_id1])
                                    cache_demon.append(last_id1)
                                    #print(org_bfs_perm_origin)
                                    last_id1_origin = int(org_bfs_perm_origin[last_id1].item())
                                    #TODO:check the logic of bfs is right
                                    atom = mol_demon_edit.GetAtomWithIdx(last_id1_origin)
                                    print('start testing atom')
                                    assert atom.GetAtomicNum() == 6
                                    assert sum([local_bond2num[x.GetBondType()] for x in atom.GetBonds()]) <= 3
                                    print('pass testing atom')
                                    for act_id in range(2):
                                        
                                        last_id2 = mol_demon_edit.AddAtom(Chem.Atom(6))
                                        print('act_id:%d, org_mol_size: %d, cur mol size: %d, last_id1: %d, last_id2:%d' % (act_id, org_mol_true_raw.GetNumAtoms(), mol_demon_edit.GetNumAtoms(), last_id1, last_id2))
                                        node_feature_cont = torch.zeros([1, self.node_dim]).cuda()
                                        node_feature_cont[0, 0] = 1.0   # carbon                                     
                                        traj_node_inputs['node_features'].append(cur_node_features.clone())  # (1, max_size_rl, self.node_dim)
                                        traj_node_inputs['adj_features'].append(cur_adj_features.clone())  # (1, self.bond_dim, max_size_rl, max_size_rl)
                                        traj_node_inputs['node_features_cont'].append(node_feature_cont)  # (1, self.node_dim)
                                        traj_node_inputs['rewards'].append(torch.full(size=(1,1), fill_value=step_cnt).cuda())  # (1, 1)
                                        traj_node_inputs['baseline_index'].append(torch.full(size=(1,), fill_value=step_cnt).long().cuda())  # (1, 1)                                        

                                        assert keep_size + act_id == last_id2

                                        cur_node_features[0, keep_size + act_id, 0] = 1.0
                                        cur_adj_features[0, :, keep_size + act_id, keep_size + act_id] = 1.0
                                        if act_id == 0:
                                            mol_demon_edit.AddBond(last_id1_origin, last_id2, local_num2bond[0])
                                        else:
                                            mol_demon_edit.AddBond(last_id1, last_id2, local_num2bond[0])
                                        edge_feature_cont = torch.zeros([1, self.bond_dim]).cuda()
                                        edge_feature_cont[0, 0] = 1.0  # single bond                                      
                                        traj_adj_inputs['node_features'].append(cur_node_features.clone())  # 1, max_size_rl, self.node_dim
                                        traj_adj_inputs['adj_features'].append(cur_adj_features.clone())  # 1, self.bond_dim, max_size_rl, max_size_rl
                                        traj_adj_inputs['edge_features_cont'].append(edge_feature_cont)  # 1, self.bond_dim
                                        traj_adj_inputs['index'].append(torch.Tensor([[last_id1, last_id2]]).long().cuda().view(1,-1)) # (1, 2)                                        
                                        traj_adj_inputs['rewards'].append(torch.full(size=(1, 1), fill_value=step_cnt).cuda())  # (1, 1)
                                        traj_adj_inputs['baseline_index'].append(torch.full(size=(1,), fill_value=step_cnt).long().cuda())  # (1)                                        
                                        cur_adj_features[0, 0, last_id1, last_id2] = 1.0
                                        cur_adj_features[0, 0, last_id2, last_id1] = 1.0                                        
                                        last_id1 = last_id2
                                        step_cnt += 1.
                                    #print('act finish')
                                    mol_demonstrated = mol_demon_edit.GetMol()
                                    s_mol_demonstrated = Chem.MolToSmiles(mol_demonstrated, isomericSmiles=True)
                                    mol = Chem.MolFromSmiles(s_mol_demonstrated)
                                    #print('mol get finish')
                                    imp_demonstrated = env.calculate_min_plogp(mol) - env.calculate_min_plogp(org_mol_true_raw)
                                    #print('cal imp finish')
                                    sim_demonstrated = reward_target_molecule_similarity(mol, org_mol_true_raw)
                                    #print('cal sim finish')
                                    if imp_demonstrated > 0 and sim_demonstrated >= 0.6:
                                        stop_sig = True
                                        flag_success = True
                                    else:
                                        stop_sig = False
                                    cur_try += 1
                                #if 1:
                                except:
                                    print('demonstration invalid occur, continue, raw is: %s' % s_raw)
                                    continue
                            if flag_success:
                                flag_use_demon = True
                                print('successfully demonstrate behavior with %d attempts' % cur_try)
                            else:
                                print('failed in demonstrate behavior')
                                continue
                                
                        else:
                            continue



                reward_valid = 2
                reward_property = 0
                reward_length = 0 
                reward_sim = 0
                #TODO: check if it is necessary to penal the small molecule, current we do not use it
                #if num_atoms < 10:
                #    reward_length -= 1.
                #elif num_atoms >= 20
                #    reward_length += 1
                #elif num_atoms >= 30:
                #    reward_length += 4
                #reward_length = 0  # current we do not use this reward
                flag_steric_strain_filter = True
                flag_zinc_molecule_filter = True

                assert mol is not None, 'mol is None...'
                final_valid = env.check_chemical_validity(mol)

                s_tmp = Chem.MolToSmiles(mol, isomericSmiles=True)
                
                try:
                    assert final_valid is True, 'warning: use valency check during generation but the final molecule is invalid!!!, \
                                    raw is %s, cur is %s' % (s_raw, s_tmp)
                except:
                    print('warning!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    print('warning!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    print('warning!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    print('warning!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    print('warning!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    print('warning: use valency check during generation but the final molecule is invalid!!!')

                if not final_valid:
                    reward_valid -= 5 
                else:
                    final_mol = env.convert_radical_electrons_to_hydrogens(mol)
                    s = Chem.MolToSmiles(final_mol, isomericSmiles=True)
                    #s_org = Chem.MolToSmiles(org_mol, isomericSmiles=True)
                    #print('orginal simile: %s' % s_org)
                    #print('orginal simile: %s' % s_org)
                    print('raw smiles: %s' % s_raw)                    
                    print('modified simile: %s' % s)

                    final_mol = Chem.MolFromSmiles(s)
                    # mol filters with negative rewards
                    if not env.steric_strain_filter(final_mol):  # passes 3D conversion, no excessive strain
                        reward_valid -= 1 #TODO: check the magnitude of this reward.
                        flag_steric_strain_filter = False
                    if not env.zinc_molecule_filter(final_mol):  # does not contain any problematic functional groups
                        reward_valid -= 1
                        flag_zinc_molecule_filter = False

                    similairty = reward_target_molecule_similarity(final_mol, org_mol_true_raw)
                    #reward_sim = similairty
                    # todo: add arg for property_type here
                    property_type = self.args.property
                    assert property_type in ['qed', 'plogp'], 'unsupported property optimization, choices are [qed, plogp]'

                    try:
                        if property_type == 'qed':
                            score = env.qed(final_mol) # value in [0, 1]

                            if self.args.reward_type == 'exp':
                                reward_property += (np.exp(score / self.args.exp_temperature) - self.args.exp_bias)

                            elif self.args.reward_type == 'linear':
                                reward_property += (score * self.args.qed_coeff)

                            if score > 0.940:
                                save_one_mol_path = os.path.join(self.args.save_path, 'good_mol_qed.txt')
                                save_one_mol(save_one_mol_path, s, cur_iter=cur_iter, score=score)

                        elif property_type == 'plogp':
                            #score = env.penalized_logp(final_mol) # unbounded value
                            #score_org = env.penalized_logp(org_mol)
                            score = env.calculate_min_plogp(final_mol)
                            score_raw = env.calculate_min_plogp(org_mol_true_raw)

                            #TODO: design stable reward....
                            if self.args.reward_type == 'exp':
                                reward_property += (np.exp(score / self.args.exp_temperature) - self.args.exp_bias) 
                            elif self.args.reward_type == 'linear':
                                reward_property += (score * self.args.plogp_coeff)
                            elif self.args.reward_type == 'imp':
                                reward_property += (score - score_raw)
                            #elif self.args.reward_type == 'exp_with_linear':
                            #    reward_property += (score * self.args.plogp_coeff + np.exp(score / self.args.exp_temperature))
                            #elif self.args.reward_type == 'neg_exp': # not working well
                            #    if score < 0:
                            #        tmp_score = -2
                            #    else:
                            #        tmp_score = np.exp(score / self.args.exp_temperature - 1.)
                            #    reward_property += (tmp_score) 
                            #score_tmp = score * self.args.plogp_coeff
                            #reward_property += (np.exp(score_tmp / self.args.exp_temperature) - 4.0)  
                            print('improvement %.4f with similarity %.4f' % (score-score_raw, similairty))
                            imp = score - score_raw
                            if self.args.not_save_demon:
                                if flag_use_demon is False:
                                    optim_dict = update_optim_dict(optim_dict, s_raw, s, imp, similairty)
                            else:
                                optim_dict = update_optim_dict(optim_dict, s_raw, s, imp, similairty)

                            if imp > 0 and similairty >= 0.6:
                                save_one_mol_path = os.path.join(self.args.save_path, 'good_mol_plogp_co.txt')
                                save_one_mol(save_one_mol_path, s, cur_iter=cur_iter, score=imp)
                    except:
                        print('generated mol does not pass env.qed/plogp')
                        #reward_property -= 2.0 
                        #TODO: check what we should do if the model do not pass qed/plogp test
                        # workaround1: add a extra penalty reward
                        # workaround2: discard the molecule.

                reward_final_total = reward_valid + reward_property + reward_length
                #reward_final_total = reward_property
                per_mol_reward.append(reward_final_total)
                per_mol_property_score.append(reward_property)

                # todo: add arg for reward decay weight here
                reward_decay = self.args.reward_decay
                print('----------------------')


                node_inputs['node_features'].append(torch.cat(traj_node_inputs['node_features'], dim=0)) #append tensor of shape (max_size_rl, max_size_rl, self.node_dim)
                node_inputs['adj_features'].append(torch.cat(traj_node_inputs['adj_features'], dim=0)) # append tensor of shape (max_size_rl, bond_dim, max_size_rl, max_size_rl)
                node_inputs['node_features_cont'].append(torch.cat(traj_node_inputs['node_features_cont'], dim=0)) # append tensor of shape (max_size_rl, 9)

                traj_node_inputs_baseline_index = torch.cat(traj_node_inputs['baseline_index'], dim=0) #(max_size_rl)
                traj_node_inputs_rewards = torch.cat(traj_node_inputs['rewards'], dim=0) # tensor of shape (max_size_rl, 1)
                traj_node_inputs_rewards[traj_node_inputs_rewards > 0] = \
                    reward_final_total * torch.pow(reward_decay, step_cnt - 1. - traj_node_inputs_rewards[traj_node_inputs_rewards > 0])
                node_inputs['rewards'].append(traj_node_inputs_rewards)  # append tensor of shape (max_size_rl, 1)                
                node_inputs['baseline_index'].append(traj_node_inputs_baseline_index)

                for ss in range(traj_node_inputs_rewards.size(0)):
                    reward_baseline[traj_node_inputs_baseline_index[ss]][0] += 1.0
                    reward_baseline[traj_node_inputs_baseline_index[ss]][1] += traj_node_inputs_rewards[ss][0]                
                

                adj_inputs['node_features'].append(torch.cat(traj_adj_inputs['node_features'], dim=0)) # (step, max_size_rl, self.node_dim)
                adj_inputs['adj_features'].append(torch.cat(traj_adj_inputs['adj_features'], dim=0)) # (step, bond_dim, max_size_rl, max_size_rl)
                adj_inputs['edge_features_cont'].append(torch.cat(traj_adj_inputs['edge_features_cont'], dim=0)) # (step, 4)
                adj_inputs['index'].append(torch.cat(traj_adj_inputs['index'], dim=0)) # (step, 2)

                traj_adj_inputs_baseline_index = torch.cat(traj_adj_inputs['baseline_index'], dim=0) #(step)                
                traj_adj_inputs_rewards = torch.cat(traj_adj_inputs['rewards'], dim=0)
                traj_adj_inputs_rewards[traj_adj_inputs_rewards > 0] = \
                    reward_final_total * torch.pow(reward_decay, step_cnt - 1. - traj_adj_inputs_rewards[traj_adj_inputs_rewards > 0])
                adj_inputs['rewards'].append(traj_adj_inputs_rewards)
                adj_inputs['baseline_index'].append(traj_adj_inputs_baseline_index)

                for ss in range(traj_adj_inputs_rewards.size(0)):
                    reward_baseline[traj_adj_inputs_baseline_index[ss]][0] += 1.0
                    reward_baseline[traj_adj_inputs_baseline_index[ss]][1] += traj_adj_inputs_rewards[ss][0]

        self.flow_core.train()
        #TODO: check whether to finetuning batch normalization
        for module in self.modules():
            if isinstance(module, torch.nn.modules.BatchNorm1d):
                module.eval()

        for i in range(reward_baseline.size(0)):
            if reward_baseline[i, 0] == 0:
                reward_baseline[i, 0] += 1.

        reward_baseline_per_step = reward_baseline[:, 1] / reward_baseline[:, 0] # (max_size_rl, )
        #TODO: check the baseline for invalid edge penalty step....
        # panelty step do not have property reward. So its magnitude may be quite different from others.

        if in_baseline is not None:
        #    #print(reward_baseline_per_step.size())
            assert in_baseline.size() == reward_baseline_per_step.size()
            reward_baseline_per_step = reward_baseline_per_step * (1. - self.args.moving_coeff) + in_baseline * self.args.moving_coeff
            print('calculating moving baseline per step')

        node_inputs_node_features = torch.cat(node_inputs['node_features'], dim=0) # (total_size, max_size_rl, 9)
        node_inputs_adj_features = torch.cat(node_inputs['adj_features'], dim=0) # (total_size, 4, max_size_rl, max_size_rl)
        node_inputs_node_features_cont = torch.cat(node_inputs['node_features_cont'], dim=0) # (total_size, 9)
        node_inputs_rewards = torch.cat(node_inputs['rewards'], dim=0).view(-1) # (total_size,)
        node_inputs_baseline_index = torch.cat(node_inputs['baseline_index'], dim=0).long() # (total_size,)
        node_inputs_baseline = torch.index_select(reward_baseline_per_step, dim=0, index=node_inputs_baseline_index) #(total_size, )

        adj_inputs_node_features = torch.cat(adj_inputs['node_features'], dim=0) # (total_size, max_size_rl, 9)
        adj_inputs_adj_features = torch.cat(adj_inputs['adj_features'], dim=0) # (total_size, 4, max_size_rl, max_size_rl)
        adj_inputs_edge_features_cont = torch.cat(adj_inputs['edge_features_cont'], dim=0) # (total_size, 4)
        adj_inputs_index = torch.cat(adj_inputs['index'], dim=0) # (total_size, 2)
        adj_inputs_rewards = torch.cat(adj_inputs['rewards'], dim=0).view(-1) # (total_size,)
        adj_inputs_baseline_index = torch.cat(adj_inputs['baseline_index'], dim=0).long() #(total_size,)
        adj_inputs_baseline = torch.index_select(reward_baseline_per_step, dim=0, index=adj_inputs_baseline_index) #(total_size, )

        if self.args.deq_type == 'random':
            #! here comes the randomness from torch cpu
            # we put the device in torch.rand so it is genereated directly on gpu
            node_inputs_node_features_cont += self.args.deq_coeff * torch.rand(
                node_inputs_node_features_cont.size(), 
                device='cuda:%d' % (node_inputs_node_features_cont.get_device()))  # (total_size, 9)
            adj_inputs_edge_features_cont += self.args.deq_coeff * torch.rand(
                adj_inputs_edge_features_cont.size(),
                device='cuda:%d' % (adj_inputs_edge_features_cont.get_device()))  # (total_size, 4)

        elif self.args.deq_type == 'variational':
            # TODO: add variational deq.
            raise ValueError('current unsupported method: %s' % self.args.deq_type)
        else:
            raise ValueError('unsupported dequantization type (%s)' % self.args.deq_type)

        if self.dp:
            node_function = self.flow_core.module.forward_rl_node
            edge_function = self.flow_core.module.forward_rl_edge

            node_function_old = self.flow_core_old.module.forward_rl_node
            edge_function_old = self.flow_core_old.module.forward_rl_edge
        else:
            node_function = self.flow_core.forward_rl_node
            edge_function = self.flow_core.forward_rl_edge
            
            node_function_old = self.flow_core_old.forward_rl_node
            edge_function_old = self.flow_core_old.forward_rl_edge

        if not self.args.split_batch:
            z_node, logdet_node = node_function(node_inputs_node_features, node_inputs_adj_features,
                                                node_inputs_node_features_cont)  # (total_step, 9), (total_step, )

            z_edge, logdet_edge = edge_function(adj_inputs_node_features, adj_inputs_adj_features,
                                                adj_inputs_edge_features_cont, adj_inputs_index) # (total_step, 4), (total_step, )
        else:
            mid_node = node_inputs_node_features.size(0) // 2
            z_node1, logdet_node1 = node_function(node_inputs_node_features[:mid_node], node_inputs_adj_features[:mid_node],
                                                node_inputs_node_features_cont[:mid_node])  # (total_step, 9), (total_step, )
            z_node2, logdet_node2 = node_function(node_inputs_node_features[mid_node:], node_inputs_adj_features[mid_node:],
                                                node_inputs_node_features_cont[mid_node:])  # (total_step, 9), (total_step, )
            z_node = torch.cat([z_node1, z_node2], dim=0)                                                                                                            
            logdet_node = torch.cat([logdet_node1, logdet_node2], dim=0)

            mid_edge = adj_inputs_node_features.size(0) // 2
            z_edge1, logdet_edge1 = edge_function(adj_inputs_node_features[:mid_edge], adj_inputs_adj_features[:mid_edge],
                                                adj_inputs_edge_features_cont[:mid_edge], adj_inputs_index[:mid_edge]) # (total_step, 4), (total_step, )
            z_edge2, logdet_edge2 = edge_function(adj_inputs_node_features[mid_edge:], adj_inputs_adj_features[mid_edge:],
                                                adj_inputs_edge_features_cont[mid_edge:], adj_inputs_index[mid_edge:]) # (total_step, 4), (total_step, )                                                            
            z_edge = torch.cat([z_edge1, z_edge2], dim=0)
            logdet_edge = torch.cat([logdet_edge1, logdet_edge2], dim=0)


        with torch.no_grad():
            if not self.args.split_batch:
                z_node_old, logdet_node_old = node_function_old(node_inputs_node_features, node_inputs_adj_features,
                                                    node_inputs_node_features_cont)  # (total_step, 9), (total_step, )

                z_edge_old, logdet_edge_old = edge_function_old(adj_inputs_node_features, adj_inputs_adj_features,
                                                    adj_inputs_edge_features_cont, adj_inputs_index) # (total_step, 4), (total_step, )
            else:
                mid_node = node_inputs_node_features.size(0) // 2
                z_node_old1, logdet_node_old1 = node_function_old(node_inputs_node_features[:mid_node], node_inputs_adj_features[:mid_node],
                                                    node_inputs_node_features_cont[:mid_node])  # (total_step, 9), (total_step, )
                z_node_old2, logdet_node_old2 = node_function_old(node_inputs_node_features[mid_node:], node_inputs_adj_features[mid_node:],
                                                    node_inputs_node_features_cont[mid_node:])  # (total_step, 9), (total_step, )
                z_node_old = torch.cat([z_node_old1, z_node_old2], dim=0)
                logdet_node_old  = torch.cat([logdet_node_old1, logdet_node_old2], dim=0)                                                                   

                mid_edge = adj_inputs_node_features.size(0) // 2
                z_edge_old1, logdet_edge_old1 = edge_function_old(adj_inputs_node_features[:mid_edge], adj_inputs_adj_features[:mid_edge],
                                                    adj_inputs_edge_features_cont[:mid_edge], adj_inputs_index[:mid_edge]) # (total_step, 4), (total_step, )
                z_edge_old2, logdet_edge_old2 = edge_function_old(adj_inputs_node_features[mid_edge:], adj_inputs_adj_features[mid_edge:],
                                                    adj_inputs_edge_features_cont[mid_edge:], adj_inputs_index[mid_edge:]) # (total_step, 4), (total_step, )                
                z_edge_old = torch.cat([z_edge_old1, z_edge_old2], dim=0)
                logdet_edge_old = torch.cat([logdet_edge_old1, logdet_edge_old2], dim=0)




        node_total_length = z_node.size(0) * float(self.node_dim)
        edge_total_length = z_edge.size(0) * float(self.bond_dim)

        #logdet_node = logdet_node - self.latent_node_length  # calculate probability of a region from probability density, minus constant has no effect on optimization
        #logdet_edge = logdet_edge - self.latent_edge_length  # calculate probability of a region from probability density, minus constant has no effect on optimization

        ll_node = -1 / 2 * (torch.log(2 * self.constant_pi) + self.prior_ln_var + torch.exp(-self.prior_ln_var) * (z_node ** 2))
        ll_node = ll_node.sum(-1)  # (B)
        ll_edge = -1 / 2 * (torch.log(2 * self.constant_pi) + self.prior_ln_var + torch.exp(-self.prior_ln_var) * (z_edge ** 2))
        ll_edge = ll_edge.sum(-1)  # (B)

        ll_node += logdet_node  # ([B])
        ll_edge += logdet_edge  # ([B])


        ll_node_old = -1 / 2 * (torch.log(2 * self.constant_pi) + self.prior_ln_var + torch.exp(-self.prior_ln_var) * (z_node_old ** 2))
        ll_node_old = ll_node_old.sum(-1)  # (B)
        ll_edge_old = -1 / 2 * (torch.log(2 * self.constant_pi) + self.prior_ln_var + torch.exp(-self.prior_ln_var) * (z_edge_old ** 2))
        ll_edge_old = ll_edge_old.sum(-1)  # (B)

        ll_node_old += logdet_node_old  # ([B])
        ll_edge_old += logdet_edge_old  # ([B])



        #lb = 1. - 0.2
        #ub = 1. + 0.2

        #ratio_node_no_clamp = torch.exp((ll_node - ll_node_old.detach()).clamp(max=50.))
        #ratio_edge_no_clamp = torch.exp((ll_edge - ll_edge_old.detach()).clamp(max=50.))
        ratio_node = torch.exp((ll_node - ll_node_old.detach()).clamp(max=10., min=-10.))
        ratio_edge = torch.exp((ll_edge - ll_edge_old.detach()).clamp(max=10., min=-10.))        

        #ratio_node = torch.clamp(ll_node - ll_node_old.detach(), math.log(lb), math.log(ub))
        #ratio_edge = torch.clamp(ll_edge - ll_edge_old.detach(), math.log(lb), math.log(ub))

        #ratio_node = torch.exp(ratio_node)
        #ratio_node = torch.exp(ll_node - ll_node_old.detach())
        if torch.isinf(ratio_node).any():
            raise RuntimeError('ratio node has inf entries')
        #ratio_edge = torch.exp(ratio_edge)
        #ratio_edge = torch.exp(ll_edge - ll_edge_old.detach())
        if torch.isinf(ratio_edge).any():
            raise RuntimeError('ratio edge has inf entries')
        if self.args.reward_type == 'imp' and self.args.no_baseline:
            advantage_node = node_inputs_rewards
            advantage_edge = adj_inputs_rewards
        else:
            advantage_node = (node_inputs_rewards - node_inputs_baseline)
            advantage_edge = (adj_inputs_rewards - adj_inputs_baseline)

        surr1_node = ratio_node * advantage_node
        surr2_node = torch.clamp(ratio_node, 1-0.2, 1+0.2) * advantage_node
        #surr2_node = ratio_node * advantage_node

        surr1_edge = ratio_edge * advantage_edge
        surr2_edge = torch.clamp(ratio_edge, 1-0.2, 1+0.2) * advantage_edge
        #surr2_edge = ratio_edge * advantage_edge


        if torch.isnan(surr1_node).any():
            raise RuntimeError('surr1 node has NaN entries')
        if torch.isnan(surr2_node).any():
            raise RuntimeError('surr2 node has NaN entries')
        if torch.isnan(surr1_edge).any():
            raise RuntimeError('surr1 edge has NaN entries')
        if torch.isnan(surr2_edge).any():
            raise RuntimeError('surr2 edge has NaN entries')                        
        #TODO: check baseline of penalty step(invalid edge)
        #TODO: check whether moving baseline is better than batch average.
        #ll_node = ll_node * (node_inputs_rewards - node_inputs_baseline)
        #ll_edge = ll_edge * (adj_inputs_rewards - adj_inputs_baseline)          

        if self.args.deq_type == 'random':
            if self.args.divide_loss:
                #print(ll_node.size())
                #print(ll_edge.size())
                #print(torch.min(surr1_node, surr2_node))
                #print(torch.min(surr1_edge, surr2_edge))
                #torch.save(torch.min(surr1_node, surr2_node), os.path.join(self.args.save_path, 'surr_node_%d' % cur_iter))
                #torch.save(torch.min(surr1_edge, surr2_edge), os.path.join(self.args.save_path, 'surr_edge_%d' % cur_iter))
                return -((torch.min(surr1_node, surr2_node).sum() + torch.min(surr1_edge, surr2_edge).sum()) / (node_total_length + edge_total_length) - 1.0), per_mol_reward, per_mol_property_score, reward_baseline_per_step, optim_dict
            else:
                # ! useless
                return -torch.sum(ll_node + ll_edge) / batch_length  # scalar




    #TODO: generate graph, finish this part
    def generate(self, temperature=0.75, mute=False, max_atoms=48, cnt=None):
        """
        inverse flow to generate molecule
        Args: 
            temp: temperature of normal distributions, we sample from (0, temp^2 * I)
        """
        generate_start_t = time()
        with torch.no_grad():
            num2bond =  {0: Chem.rdchem.BondType.SINGLE, 1: Chem.rdchem.BondType.DOUBLE, 2: Chem.rdchem.BondType.TRIPLE}
            num2bond_symbol =  {0: '=', 1: '==', 2: '==='}
            # [6, 7, 8, 9, 15, 16, 17, 35, 53, 0]
            num2atom = {0: 6, 1: 7, 2: 8, 3: 9, 4:15, 5:16, 6:17, 7:35, 8:53}
            num2symbol = {0: 'C', 1: 'N', 2: 'O', 3: 'F', 4:'P', 5:'S', 6:'Cl', 7:'Br', 8:'I'}


            prior_node_dist = torch.distributions.normal.Normal(torch.zeros([self.node_dim]).cuda(), 
                                        temperature * torch.ones([self.node_dim]).cuda())
            prior_edge_dist = torch.distributions.normal.Normal(torch.zeros([self.bond_dim]).cuda(), 
                                        temperature * torch.ones([self.bond_dim]).cuda())

            cur_node_features = torch.zeros([1, max_atoms, self.node_dim]).cuda()
            cur_adj_features = torch.zeros([1, self.bond_dim, max_atoms, max_atoms]).cuda()

            rw_mol = Chem.RWMol() # editable mol
            mol = None
            #mol_size = mol.GetNumAtoms()

            is_continue = True
            total_resample = 0
            each_node_resample = np.zeros([max_atoms])
            for i in range(max_atoms):
                if not is_continue:
                    break
                if i < self.edge_unroll:
                    edge_total = i # edge to sample for current node
                    start = 0
                else:
                    edge_total = self.edge_unroll
                    start = i - self.edge_unroll
                # first generate node
                ## reverse flow
                latent_node = prior_node_dist.sample().view(1, -1) #(1, 9)
                if self.dp:
                    latent_node = self.flow_core.module.reverse(cur_node_features, cur_adj_features, 
                                            latent_node, mode=0).view(-1) # (9, )
                else:
                    latent_node = self.flow_core.reverse(cur_node_features, cur_adj_features, 
                                            latent_node, mode=0).view(-1) # (9, )
                ## node/adj postprocessing
                #print(latent_node.shape) #(38, 9)
                feature_id = torch.argmax(latent_node).item()
                #print(num2symbol[feature_id])
                cur_node_features[0, i, feature_id] = 1.0
                cur_adj_features[0, :, i, i] = 1.0
                rw_mol.AddAtom(Chem.Atom(num2atom[feature_id]))
                

                # then generate edges
                if i == 0:
                    is_connect = True
                else:
                    is_connect = False
                #cur_mol_size = mol.GetNumAtoms
                for j in range(edge_total):
                    valid = False
                    resample_edge = 0
                    invalid_bond_type_set = set()
                    while not valid:
                        #TODO: add cache. Some atom can not get the right edge type and is stuck in the loop
                        #TODO: add cache. invalid bond set
                        if len(invalid_bond_type_set) < 3 and resample_edge <= 50: # haven't sampled all possible bond type or is not stuck in the loop
                            latent_edge = prior_edge_dist.sample().view(1, -1) #(1, 4)
                            if self.dp:
                                latent_edge = self.flow_core.module.reverse(cur_node_features, cur_adj_features, latent_edge, 
                                            mode=1, edge_index=torch.Tensor([[j + start, i]]).long().cuda()).view(-1) #(4, )
                            else:
                                latent_edge = self.flow_core.reverse(cur_node_features, cur_adj_features, latent_edge, 
                                            mode=1, edge_index=torch.Tensor([[j + start, i]]).long().cuda()).view(-1) #(4, )
                            edge_discrete_id = torch.argmax(latent_edge).item()
                        else:
                            if not mute:
                                print('have tried all possible bond type, use virtual bond.')
                            assert resample_edge > 50 or len(invalid_bond_type_set) == 3
                            edge_discrete_id = 3 # we have no choice but to choose not to add edge between (i, j+start)
                        cur_adj_features[0, edge_discrete_id, i, j + start] = 1.0
                        cur_adj_features[0, edge_discrete_id, j + start, i] = 1.0
                        if edge_discrete_id == 3: # virtual edge
                            valid = True
                        else: #single/double/triple bond
                            rw_mol.AddBond(i, j + start, num2bond[edge_discrete_id])                                                   
                            valid = env.check_valency(rw_mol)
                            if valid:
                                is_connect = True
                                #print(num2bond_symbol[edge_discrete_id])
                            else: #backtrack
                                rw_mol.RemoveBond(i, j + start)
                                cur_adj_features[0, edge_discrete_id, i, j + start] = 0.0
                                cur_adj_features[0, edge_discrete_id, j + start, i] = 0.0
                                total_resample += 1.0
                                each_node_resample[i] += 1.0
                                resample_edge += 1

                                invalid_bond_type_set.add(edge_discrete_id)
                if is_connect: # new generated node has at least one bond with previous node, do not stop generation, backup mol from rw_mol to mol
                    is_continue = True
                    mol = rw_mol.GetMol()
                
                else:
                    is_continue = False

            #mol = rw_mol.GetMol() # mol backup
            assert mol is not None, 'mol is None...'

            #final_valid = check_valency(mol)
            final_valid = env.check_chemical_validity(mol)            
            assert final_valid is True, 'warning: use valency check during generation but the final molecule is invalid!!!'            

            final_mol = env.convert_radical_electrons_to_hydrogens(mol)
            smiles = Chem.MolToSmiles(final_mol, isomericSmiles=True)
            assert '.' not in smiles, 'warning: use is_connect to check stop action, but the final molecule is disconnected!!!'

            final_mol = Chem.MolFromSmiles(smiles)


            #mol = convert_radical_electrons_to_hydrogens(mol)
            num_atoms = final_mol.GetNumAtoms()
            num_bonds = final_mol.GetNumBonds()

            pure_valid = 0
            if total_resample == 0:
                pure_valid = 1.0
            if not mute:
                cnt = str(cnt) if cnt is not None else ''
                print('smiles%s: %s | #atoms: %d | #bonds: %d | #resample: %.5f | time: %.5f |' % (cnt, smiles, num_atoms, num_bonds, total_resample, time()-generate_start_t))
            return smiles, pure_valid, num_atoms
    

    #TODO: batched generation, current useless
    def batch_generate(self, batch_size=100, temperature=0.75, mute=False):
        """
        inverse flow to generate one batch molecules
        Args: 
            temp: temperature of normal distributions, we sample from (0, temp^2 * I)
            mute: do not print some messages
        Returns:
            generated mol represented by smiles, valid rate (without check)
        """

        generate_start_t = time()
        with torch.no_grad():
            num2bond =  {0: Chem.rdchem.BondType.SINGLE, 1: Chem.rdchem.BondType.DOUBLE, 2: Chem.rdchem.BondType.TRIPLE}
            num2bond_symbol =  {0: '=', 1: '==', 2: '==='}
            # [6, 7, 8, 9, 15, 16, 17, 35, 53, 0]
            num2atom = {0: 6, 1: 7, 2: 8, 3: 9, 4:15, 5:16, 6:17, 7:35, 8:53}
            num2symbol = {0: 'C', 1: 'N', 2: 'O', 3: 'F', 4:'P', 5:'S', 6:'Cl', 7:'Br', 8:'I'}


            prior_node_dist = torch.distributions.normal.Normal(torch.zeros([self.node_dim]).cuda(), 
                                        temperature * torch.ones([self.node_dim]).cuda())
            prior_edge_dist = torch.distributions.normal.Normal(torch.zeros([self.bond_dim]).cuda(), 
                                        temperature * torch.ones([self.bond_dim]).cuda())

            cur_node_features = torch.zeros([batch_size, self.max_size, self.node_dim]).cuda()
            cur_adj_features = torch.zeros([batch_size, self.bond_dim, self.max_size, self.max_size]).cuda()

            rw_mol = Chem.RWMol() # editable mol
            mol = None
            #mol_size = mol.GetNumAtoms()

            is_continue = [True] * batch_size
            total_resample = [0] * batch_size
            each_node_resample = np.zeros([batch_size,self.max_size])
            for i in range(self.max_size):
                if not is_continue:
                    break
                if i < self.edge_unroll:
                    edge_total = i # edge to sample for current node
                    start = 0
                else:
                    edge_total = self.edge_unroll
                    start = i - self.edge_unroll
                # first generate node
                ## reverse flow
                a.sample(sample_shape=torch.Size([20]))
                latent_node = prior_node_dist.sample().view(1, -1) #(1, 9)
                if self.dp:
                    latent_node = self.flow_core.module.reverse(cur_node_features, cur_adj_features, 
                                            latent_node, mode=0).view(-1) # (9, )
                else:
                    latent_node = self.flow_core.reverse(cur_node_features, cur_adj_features, 
                                            latent_node, mode=0).view(-1) # (9, )
                ## node/adj postprocessing
                #print(latent_node.shape) #(38, 9)
                feature_id = torch.argmax(latent_node).item()
                #print(num2symbol[feature_id])
                cur_node_features[0, i, feature_id] = 1.0
                cur_adj_features[0, :, i, i] = 1.0
                rw_mol.AddAtom(Chem.Atom(num2atom[feature_id]))
                

                # then generate edges
                if i == 0:
                    is_connect = True
                else:
                    is_connect = False
                #cur_mol_size = mol.GetNumAtoms
                for j in range(edge_total):
                    valid = False
                    resample_edge = 0
                    invalid_bond_type_set = set()
                    while not valid:
                        #TODO: add cache. Some atom can not get the right edge type and is stuck in the loop
                        #TODO: add cache. invalid bond set
                        if len(invalid_bond_type_set) < 3 and resample_edge <= 50: # haven't sampled all possible bond type or is not stuck in the loop
                            latent_edge = prior_edge_dist.sample().view(1, -1) #(1, 4)
                            if self.dp:
                                latent_edge = self.flow_core.module.reverse(cur_node_features, cur_adj_features, latent_edge, 
                                            mode=1, edge_index=torch.Tensor([[j + start, i]]).long().cuda()).view(-1) #(4, )
                            else:
                                latent_edge = self.flow_core(cur_node_features, cur_adj_features, latent_edge, 
                                            mode=1, edge_index=torch.Tensor([[j + start, i]]).long().cuda()).view(-1) #(4, )
                            edge_discrete_id = torch.argmax(latent_edge).item()
                        else:
                            if not mute:
                                print('have tried all possible bond type, use virtual bond.')
                            assert resample_edge > 50 or len(invalid_bond_type_set) == 3
                            edge_discrete_id = 3 # we have no choice but to choose not to add edge between (i, j+start)
                        cur_adj_features[0, edge_discrete_id, i, j + start] = 1.0
                        cur_adj_features[0, edge_discrete_id, j + start, i] = 1.0
                        if edge_discrete_id == 3: # virtual edge
                            valid = True
                        else: #single/double/triple bond
                            rw_mol.AddBond(i, j + start, num2bond[edge_discrete_id])                                                   
                            #TODO: check valency 
                            valid = env.check_valency(rw_mol)
                            if valid:
                                is_connect = True
                                #print(num2bond_symbol[edge_discrete_id])
                            else: #backtrack
                                rw_mol.RemoveBond(i, j + start)
                                cur_adj_features[0, edge_discrete_id, i, j + start] = 0.0
                                cur_adj_features[0, edge_discrete_id, j + start, i] = 0.0
                                total_resample += 1.0
                                each_node_resample[i] += 1.0
                                resample_edge += 1

                                invalid_bond_type_set.add(edge_discrete_id)
                if is_connect: # new generated node has at least one bond with previous node, do not stop generation, backup mol from rw_mol to mol
                    is_continue = True
                    mol = rw_mol.GetMol()
                
                else:
                    is_continue = False

            #mol = rw_mol.GetMol() # mol backup
            assert mol is not None, 'mol is None...'
            mol = env.convert_radical_electrons_to_hydrogens(mol)
            num_atoms = mol.GetNumAtoms()
            num_bonds = mol.GetNumBonds()

            smiles = Chem.MolToSmiles(mol)
            assert '.' not in smiles, 'warning: use is_connect to check stop action, but the final molecule is disconnected!!!'

            final_valid = env.check_valency(mol)
            assert final_valid is True, 'warning: use valency check during generation but the final molecule is invalid!!!'

            pure_valid = 0
            if total_resample == 0:
                pure_valid = 1.0
            if not mute:
                print('smiles: %s | #atoms: %d | #bonds: %d | #resample: %.5f | time: %.5f |' % (smiles, num_atoms, num_bonds, total_resample, time()-generate_start_t))
            return smiles, pure_valid    

    def initialize_masks(self, max_node_unroll=38, max_edge_unroll=12):
        """
        Args:
            max node unroll: maximal number of nodes in molecules to be generated (default: 38)
            max edge unroll: maximal number of edges to predict for each generated nodes (default: 12, calculated from zink250K data)
        Returns:
            node_masks: node mask for each step
            adj_masks: adjacency mask for each step
            is_node_update_mask: 1 indicate this step is for updating node features
            flow_core_edge_mask: get the distributions we want to model in adjacency matrix
        """
        num_masks = int(max_node_unroll + (max_edge_unroll - 1) * max_edge_unroll / 2 + (max_node_unroll - max_edge_unroll) * (max_edge_unroll))
        num_mask_edge = int(num_masks - max_node_unroll)

        node_masks1 = torch.zeros([max_node_unroll, max_node_unroll]).byte()
        adj_masks1 = torch.zeros([max_node_unroll, max_node_unroll, max_node_unroll]).byte()
        node_masks2 = torch.zeros([num_mask_edge, max_node_unroll]).byte()
        adj_masks2 = torch.zeros([num_mask_edge, max_node_unroll, max_node_unroll]).byte()        
        #is_node_update_masks = torch.zeros([num_masks]).byte()

        link_prediction_index = torch.zeros([num_mask_edge, 2]).long()


        flow_core_edge_masks = torch.zeros([max_node_unroll, max_node_unroll]).byte()

        #masks_edge = dict()
        cnt = 0
        cnt_node = 0
        cnt_edge = 0
        for i in range(max_node_unroll):
            node_masks1[cnt_node][:i] = 1
            adj_masks1[cnt_node][:i, :i] = 1
            #is_node_update_masks[cnt] = 1
            cnt += 1
            cnt_node += 1

            edge_total = 0
            if i < max_edge_unroll:
                start = 0
                edge_total = i
            else:
                start = i - max_edge_unroll
                edge_total = max_edge_unroll
            for j in range(edge_total):
                if j == 0:
                    node_masks2[cnt_edge][:i+1] = 1
                    adj_masks2[cnt_edge] = adj_masks1[cnt_node-1].clone()
                    adj_masks2[cnt_edge][i,i] = 1
                else:
                    node_masks2[cnt_edge][:i+1] = 1
                    adj_masks2[cnt_edge] = adj_masks2[cnt_edge-1].clone()
                    adj_masks2[cnt_edge][i, start + j -1] = 1
                    adj_masks2[cnt_edge][start + j -1, i] = 1
                cnt += 1
                cnt_edge += 1
        assert cnt == num_masks, 'masks cnt wrong'
        assert cnt_node == max_node_unroll, 'node masks cnt wrong'
        assert cnt_edge == num_mask_edge, 'edge masks cnt wrong'

    
        cnt = 0
        for i in range(max_node_unroll):
            if i < max_edge_unroll:
                start = 0
                edge_total = i
            else:
                start = i - max_edge_unroll
                edge_total = max_edge_unroll
        
            for j in range(edge_total):
                link_prediction_index[cnt][0] = start + j
                link_prediction_index[cnt][1] = i
                cnt += 1
        assert cnt == num_mask_edge, 'edge mask initialize fail'


        for i in range(max_node_unroll):
            if i == 0:
                continue
            if i < max_edge_unroll:
                start = 0
                end = i
            else:
                start = i - max_edge_unroll
                end = i 
            flow_core_edge_masks[i][start:end] = 1

        node_masks = torch.cat((node_masks1, node_masks2), dim=0)
        adj_masks = torch.cat((adj_masks1, adj_masks2), dim=0)

        node_masks = nn.Parameter(node_masks, requires_grad=False)
        adj_masks = nn.Parameter(adj_masks, requires_grad=False)
        link_prediction_index = nn.Parameter(link_prediction_index, requires_grad=False)
        flow_core_edge_masks = nn.Parameter(flow_core_edge_masks, requires_grad=False)
        
        return node_masks, adj_masks, link_prediction_index, flow_core_edge_masks




    def log_prob(self, z, logdet, deq_logp=None, deq_logdet=None):
          
        #TODO: check multivariate gaussian log_prob formula
        logdet[0] = logdet[0] - self.latent_node_length # calculate probability of a region from probability density, minus constant has no effect on optimization
        logdet[1] = logdet[1] - self.latent_edge_length # calculate probability of a region from probability density, minus constant has no effect on optimization

        ll_node = -1/2 * (torch.log(2 * self.constant_pi) + self.prior_ln_var + torch.exp(-self.prior_ln_var) * (z[0]**2))
        ll_node = ll_node.sum(-1) # (B)

        ll_edge = -1/2 * (torch.log(2 * self.constant_pi) + self.prior_ln_var + torch.exp(-self.prior_ln_var) * (z[1]**2))
        ll_edge = ll_edge.sum(-1) # (B)



        ll_node += logdet[0] #([B])
        ll_edge += logdet[1] #([B])
        
        if self.args.deq_type == 'random':
            if self.args.divide_loss:
                return -(torch.mean(ll_node + ll_edge) / (self.latent_edge_length + self.latent_node_length))
            else:
                #! useless
                return -torch.mean(ll_node + ll_edge) # scalar

        elif self.args.deq_type == 'variational':
            #TODO: finish this part
            assert deq_logp is not None and deq_logdet is not None, 'variational dequantization requires deq_logp/deq_logdet'
            ll_deq_node = deq_logp[0] - deq_logdet[0] #()
            #print(ll_deq_node.size())
            ll_deq_edge = deq_logp[1] - deq_logdet[1]
            #print(ll_deq_edge.size())
            return (torch.mean(ll_node), torch.mean(ll_edge), torch.mean(ll_deq_node), torch.mean(ll_deq_edge))

        
        

