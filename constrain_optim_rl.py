#coding: utf-8
'''
Anonymous author
'''

from time import time
import argparse
import numpy as np
import math
import os
import sys
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from utils import *

from model_rl import GraphFlowModel

from dataloader import ConstrainOptim_Zink800, DataIterator
import environment as env

from rdkit import DataStructs
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import AllChem

def adjust_learning_rate(optimizer, cur_iter, init_lr, warm_up_step):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # if warm up step is 0, no warm up actually.
    if cur_iter < warm_up_step:
        lr = init_lr * (1. / warm_up_step + 1. / warm_up_step * cur_iter)  # [0.1lr, 0.2lr, 0.3lr, ..... 1lr]
    else:
        lr = init_lr
        return lr
    #lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def target_molecule_similarity(mol, target, radius=2, nBits=2048,
                                      useChirality=True):
    """
    taken from GCPN's implementation
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



def save_model(model, optimizer, args, var_list, epoch=None):
    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as f:
        json.dump(argparse_dict, f)

    epoch = str(epoch) if epoch is not None else ''
    latest_save_path = os.path.join(args.save_path, 'checkpoint')
    final_save_path = os.path.join(args.save_path, 'checkpoint%s' % epoch)
    torch.save({
        **var_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        final_save_path
    )

    # save twice to maintain a latest checkpoint
    torch.save({
        **var_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        latest_save_path
    )    

def restore_model(model, args, epoch=None):
    if epoch is None:
        restore_path = os.path.join(args.save_path, 'checkpoint')
        print('load from the latest checkpoint')
    else:
        restore_path = os.path.join(args.save_path, 'checkpoint%s' % str(epoch))
        print('load from checkpoint%s' % str(epoch))

    checkpoint = torch.load(restore_path)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print('load model state dict, strict is set to False!')




def read_molecules(path, read_all_smiles=False):
    print('reading data from %s' % path)
    node_features = np.load(path + '_node_features.npy')
    adj_features = np.load(path + '_adj_features.npy')
    mol_sizes = np.load(path + '_mol_sizes.npy')

    f = open(path + '_config.txt', 'r')
    data_config = eval(f.read())
    f.close()

    if read_all_smiles:
        fp = open(path + '_raw_smiles.smi')
        all_smiles = []
        for smiles in fp:
            all_smiles.append(smiles.strip())
        fp.close()
    else:
        all_smiles = None
    return node_features, adj_features, mol_sizes, data_config, all_smiles

def read_smi_plogp(path):
    node_features = np.load(path + '_node_features.npy')
    adj_features = np.load(path + '_adj_features.npy')
    mol_sizes = np.load(path + '_mol_sizes.npy')


    f = open(path + '_config.txt', 'r')
    data_config = eval(f.read())
    f.close()

    fp = open(path + '.smi_logp')
    all_smiles = []
    all_logps = []
    for line in fp:
        line = line.strip().split(',')
        all_logps.append(float(line[0]))
        all_smiles.append(line[1])
    fp.close()
    return node_features, adj_features, mol_sizes, data_config, all_smiles, all_logps



class Trainer(object):
    def __init__(self, dataloader, data_config, args, all_train_smiles=None):
        if args.reinforce_fintune:
            self.dataloader = DataIterator(dataloader)
        else:
            self.dataloader = dataloader
        self.data_config = data_config
        self.args = args
        self.all_train_smiles = all_train_smiles

        self.max_size = self.data_config['max_size']
        self.node_dim = self.data_config['node_dim'] - 1 # exclude padding dim.
        self.bond_dim = self.data_config['bond_dim']
       
        
        self._model = GraphFlowModel(self.max_size, self.node_dim, self.bond_dim, self.args.edge_unroll, self.args)
        self._optimizer_rl = optim.Adam(filter(lambda p: p.requires_grad, self._model.flow_core.parameters()),
                        lr=self.args.lr, weight_decay=self.args.weight_decay)

        self.best_reward = -100.0
        self.start_iter = 0
        if self.args.cuda:
            self._model = self._model.cuda()
        print(self._model.state_dict().keys()) # check the key consistency. 
    

    def initialize_from_checkpoint(self, gen=False):
        
        checkpoint = torch.load(self.args.init_checkpoint)
        print(checkpoint['model_state_dict'].keys()) # check the key consistency     
        self._model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        if 'reinforce' in self.args.init_checkpoint:
            print('loading optim state from reinforce checkpoint...')
            self._optimizer_rl.load_state_dict(checkpoint['optimizer_state_dict'])
            self.best_reward = checkpoint['best_reward']
            self.start_iter = checkpoint['cur_iter'] + 1        

        print('initialize from %s Done!' % self.args.init_checkpoint)


    def reinforce(self, mol_out_dir=None):        
        t_total = time()
        total_loss = []
        total_reward = []
        total_score = []
        total_optim_dict = {}

        best_reward = self.best_reward
        start_iter = self.start_iter
        #moving_baseline = np.zeros([self.args.max_size_rl])
        moving_baseline = None
        print('start finetuning model(reinforce)')
        for cur_iter in range(self.args.reinforce_iters):
            if cur_iter == 0:
                iter_loss, iter_reward, iter_score, moving_baseline, optim_dict = self.reinforce_one_iter(cur_iter + start_iter, in_baseline=None)
            else:
                iter_loss, iter_reward, iter_score, moving_baseline, optim_dict = self.reinforce_one_iter(cur_iter + start_iter, in_baseline=moving_baseline)

            total_optim_dict = update_total_optim_dict(total_optim_dict, optim_dict)

            # save the updated dict
            f_tmp = open(os.path.join(self.args.save_path, 'optim_dict.txt'), 'w')
            f_tmp.write(str(total_optim_dict)) 
            f_tmp.close
             
            total_loss.append(iter_loss)
            total_reward.append(iter_reward)
            total_score.append(iter_score)
            save_one_reward(os.path.join(self.args.save_path, 'iter_rewards.txt'), iter_reward, iter_score, iter_loss, cur_iter + start_iter) # append the iter reward to file
            print(moving_baseline)

            if iter_reward > best_reward:
                best_reward = iter_reward
                if self.args.save:
                    var_list = {'cur_iter': cur_iter + start_iter,
                                'best_reward': best_reward,
                               }
                    save_model(self._model, self._optimizer_rl, self.args, var_list, epoch=cur_iter + start_iter)

        print("Finetuning(Reinforce) Finished!")
        print("Total time elapsed: {:.4f}s".format(time() - t_total))


    def reinforce_one_iter(self, iter_cnt, in_baseline=None):
        t_start = time()
        #self._model.train() we will manually set train/eval mode in self._model.reinforce_forward()
        #if iter_cnt % self.args.accumulate_iters == 0:
        self._optimizer_rl.zero_grad()
        batch_data = next(self.dataloader)
        mol_xs = batch_data['node']
        mol_adjs = batch_data['adj']
        mol_sizes = batch_data['mol_size']
        bfs_perm_origin = batch_data['bfs_perm_origin']
        raw_smiles = batch_data['raw_smile']

    
        loss, per_mol_reward, per_mol_property_score, out_baseline, optim_dict = self._model.reinforce_forward_constrained_optim(
                                                mol_xs=mol_xs, mol_adjs=mol_adjs, mol_sizes=mol_sizes, 
                                                modify_size=args.modify_size, bfs_perm_origin=bfs_perm_origin, 
                                                temperature=self.args.rl_sample_temperature, raw_smiles=raw_smiles,
                                                max_size_rl=self.args.max_size_rl, batch_size=self.args.batch_size, 
                                                in_baseline=in_baseline, cur_iter=iter_cnt)

        num_mol = len(per_mol_reward)
        avg_reward = sum(per_mol_reward) / num_mol
        avg_score = sum(per_mol_property_score) / num_mol
        max_cur_reward = max(per_mol_reward)
        max_cur_score = max(per_mol_property_score)
        loss.backward()
        #if (iter_cnt + 1) % self.args.accumulate_iters == 0:
        #    print('update parameter at iter %d' % iter_cnt)
        nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self._model.flow_core.parameters()), 1.0)
        cur_lr = adjust_learning_rate(self._optimizer_rl, iter_cnt, self.args.lr, self.args.warm_up)
        self._optimizer_rl.step()
        if self.args.lr_decay:
            self._scheduler.step(avg_reward)

        print('Iter: {: d}, num_mol {:d}, loss {:5.5f}, lr {:5.5f}, reward {:5.5f}, score {:5.5f}, max_reward {:5.5f}, max_score {:5.5f}, iter time {:.5f}'.format(iter_cnt, 
                                    num_mol, loss.item(), cur_lr, 
                                    avg_reward, avg_score, max_cur_reward, max_cur_score, time()-t_start))    
        return loss.item(), avg_reward, avg_score, out_baseline, optim_dict




    def constrained_optimize(self):
        all_best_smile = []
        all_best_score = []
        #max_cnt = 1
        cur_cnt = 0
        data_len = len(self.dataloader)
        repeat_time = self.args.repeat_time
        min_optim_time = 25 # optimize one molecule for at least 10 times
        optim_success_dict = {}
        #for batch_cnt, batch_data in enumerate(self.dataloader):
        ranges=[self.args.optim_start, self.args.optim_end]        
        assert len(self.dataloader) == 800
        for batch_cnt in range(data_len):
            best_smile_all = ['Not Found', 'Not Found', 'Not Found', 'Not Found']
            best_score_all = [-100., -100., -100., -100.]
            final_sim_all = [-1., -1., -1., -1.]
            if batch_cnt < ranges[0] or batch_cnt >= ranges[1]:
                continue
            #if cur_cnt >= max_cnt:
            #    break            
            #if batch_cnt in [0]:
            #    continue
            for cur_iter in range(repeat_time):
                batch_data = self.dataloader[batch_cnt] # dataloader is dataset object

                #inp_node_features = batch_data['node'] #(1, N, node_dim)
                inp_node_features = batch_data['node'].unsqueeze(0) #(1, N, node_dim)

                #inp_adj_features = batch_data['adj'] #(1, 4, N, N)              
                inp_adj_features = batch_data['adj'].unsqueeze(0) #(1, 4, N, N)              

                raw_smile = batch_data['raw_smile']  #(1)
                mol_size = batch_data['mol_size']
                bfs_perm_origin = batch_data['bfs_perm_origin']
                bfs_perm_origin = torch.Tensor(bfs_perm_origin)

                if raw_smile not in optim_success_dict:
                    optim_success_dict[raw_smile] = [0, -1] #(try_time, imp)
                if optim_success_dict[raw_smile][0] > min_optim_time and optim_success_dict[raw_smile][1] > 0: # reach min time and imp is positive
                    continue # not optimize this one

                plogp = batch_data['plogp'] #(1)
                print(raw_smile, plogp)
                #print(inp_node_features)
                #print(inp_adj_features)
                best_smile0246, best_score0246, final_sim0246 = self.constrained_optimize_one_mol_rl(inp_adj_features, 
                                                                    inp_node_features, raw_smile, mol_size, bfs_perm_origin, cur_time=cur_iter)
                if best_score0246[0] > best_score_all[0]:
                    best_score_all[0] = best_score0246[0]
                    best_smile_all[0] = best_smile0246[0]
                    final_sim_all[0] = final_sim0246[0]

                if best_score0246[1] > best_score_all[1]:
                    best_score_all[1] = best_score0246[1]
                    best_smile_all[1] = best_smile0246[1]
                    final_sim_all[1] = final_sim0246[1]

                if best_score0246[2] > best_score_all[2]:
                    best_score_all[2] = best_score0246[2]
                    best_smile_all[2] = best_smile0246[2]
                    final_sim_all[2] = final_sim0246[2]
                    
                if best_score0246[3] > best_score_all[3]:
                    best_score_all[3] = best_score0246[3]
                    best_smile_all[3] = best_smile0246[3]
                    final_sim_all[3] = final_sim0246[3]
                if best_score_all[3] > 0: #imp > 0
                    optim_success_dict[raw_smile][1] = best_score_all[3]
                optim_success_dict[raw_smile][0] += 1 # try time + 1

            final_cg_save_path = os.path.join(self.args.optimized_save_path, 'm%dre%d' % ( 
                                                                self.args.modify_size, self.args.repeat_time))
            if not os.path.exists(final_cg_save_path):
                os.makedirs(final_cg_save_path)            
            save_one_optimized_molecule(final_cg_save_path, 
                                        raw_smile, best_smile_all, 
                                        best_score_all, cur_iter=batch_cnt, 
                                        ranges=ranges, sim=final_sim_all)
            cur_cnt += 1

        print('finish optimized %d molecules from %d to %d!' % (ranges[1] - ranges[0], ranges[0], ranges[1]))
        return 1

    def constrained_optimize_one_mol_rl(self, adj, x, org_smile, mol_size, bfs_perm_origin, cur_time):
        """
        direction: score ascent direction
        adj: adjacent matrix of origin mol (1, 4, N, N)
        x: node feature of origin mol (1, N, 9)
        """


        self._model.eval()


        best_smile0 = None
        best_smile2 = None
        best_smile4 = None
        best_smile6 = None
        best_imp0 = -100.
        best_imp2 = -100.
        best_imp4 = -100.
        best_imp6 = -100.
        final_sim0 = -1.
        final_sim2 = -1.
        final_sim4 = -1.
        final_sim6 = -1.

        #    def reinforce_constrained_optim_one_mol(self, x, adj, mol_size, modify_size, raw_smile, bfs_perm_origin, 
        #                                            temperature=0.75, mute=False, batch_size=1, max_size_rl=38):
        mol_org = Chem.MolFromSmiles(org_smile)
        mol_org_size = mol_org.GetNumAtoms()
        assert mol_org_size == mol_size
        print('%d: ****** %s ******' % (cur_time, org_smile))

        cur_mol_smiles, cur_mol_imps, cur_mol_sims = self._model.reinforce_constrained_optim_one_mol(x, adj, mol_size, self.args.modify_size,
                                                                        org_smile, bfs_perm_origin, max_size_rl=self.args.max_size_rl)
        num_success = len(cur_mol_imps)
        for i in range(num_success):
            cur_smile = cur_mol_smiles[i]
            cur_imp = cur_mol_imps[i]
            cur_sim = cur_mol_sims[i]
            assert cur_imp > 0
            if cur_sim > 0:
                if cur_imp > best_imp0:
                    best_smile0 = cur_smile
                    best_imp0 = cur_imp
                    final_sim0 = cur_sim
            if cur_sim > 0.2:
                if cur_imp > best_imp2:
                    best_smile2 = cur_smile
                    best_imp2 = cur_imp
                    final_sim2 = cur_sim
            if cur_sim > 0.4:
                if cur_imp > best_imp4:
                    best_smile4 = cur_smile
                    best_imp4 = cur_imp
                    final_sim4 = cur_sim
            if cur_sim > 0.6:
                if cur_imp > best_imp6:
                    best_smile6 = cur_smile
                    best_imp6 = cur_imp
                    final_sim6 = cur_sim                    


        return [best_smile0, best_smile2, best_smile4, best_smile6], [best_imp0, best_imp2, best_imp4, best_imp6], [final_sim0, final_sim2, final_sim4, final_sim6]



        





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GraphFlow model')

    # ******data args******
    parser.add_argument('--dataset', type=str, default='zinc250k', help='dataset')
    parser.add_argument('--path', type=str, help='path of dataset', required=True)


    parser.add_argument('--batch_size', type=int, default=64, help='batch_size.')
    parser.add_argument('--edge_unroll', type=int, default=12, help='max edge to model for each node in bfs order.')
    parser.add_argument('--shuffle', action='store_true', default=False, help='shuffle data for each epoch')
    parser.add_argument('--num_workers', type=int, default=10, help='num works to generate data.')

    # ******model args******
    parser.add_argument('--name', type=str, default='base', help='model name, crucial for test and checkpoint initialization')
    parser.add_argument('--deq_type', type=str, default='random', help='dequantization methods.')
    parser.add_argument('--deq_coeff', type=float, default=0.9, help='dequantization coefficient.(only for deq_type random)')
    parser.add_argument('--num_flow_layer', type=int, default=6, help='num of affine transformation layer in each timestep')
    parser.add_argument('--gcn_layer', type=int, default=3, help='num of rgcn layers')
    #TODO: Disentangle num of hidden units for gcn layer, st net layer.
    parser.add_argument('--nhid', type=int, default=128, help='num of hidden units of gcn')
    parser.add_argument('--nout', type=int, default=128, help='num of out units of gcn')

    parser.add_argument('--st_type', type=str, default='sigmoid', help='architecture of st net, choice: [sigmoid, exp, softplus, spine]')

    # ******for sigmoid st net only ******
    parser.add_argument('--sigmoid_shift', type=float, default=2.0, help='sigmoid shift on s.')

    # ******for exp st net only ******

    # ******for softplus st net only ******

    # ******optimization args******
    parser.add_argument('--train', action='store_true', default=False, help='do training.')
    parser.add_argument('--save', action='store_true', default=False, help='Save model.')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--learn_prior', action='store_true', default=False, help='learn log-var of gaussian prior.')

    parser.add_argument('--seed', type=int, default=2019, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--is_bn', action='store_true', default=False, help='batch norm on node embeddings.')
    parser.add_argument('--is_bn_before', action='store_true', default=False, help='batch norm on node embeddings on st-net input.')
    parser.add_argument('--scale_weight_norm', action='store_true', default=False, help='apply weight norm on scale factor.')
    parser.add_argument('--divide_loss', action='store_true', default=False, help='divide loss by length of latent.')
    parser.add_argument('--init_checkpoint', type=str, default=None, help='initialize from a checkpoint, if None, do not restore')

    parser.add_argument('--show_loss_step', type=int, default=100)

    # ******generation args******
    parser.add_argument('--temperature', type=float, default=0.75, help='temperature for normal distribution')
    parser.add_argument('--min_atoms', type=int, default=5, help='minimum #atoms of generated mol, otherwise the mol is simply discarded')
    parser.add_argument('--max_atoms', type=int, default=48, help='maximum #atoms of generated mol')    
    parser.add_argument('--gen_num', type=int, default=100, help='num of molecules to generate on each call to train.generate')
    parser.add_argument('--gen', action='store_true', default=False, help='generate')
    parser.add_argument('--gen_out_path', type=str, help='output path for generated mol')

    # ******constrained optimziation args******
    parser.add_argument('--all_save_prefix', type=str, default='./', help='path of save prefix')

    parser.add_argument('--co_rl', action='store_true', default=False, help='generate')
    
    parser.add_argument('--optim_start', type=int, default=0, help='num of molecules to generate on each call to train.generate')
    parser.add_argument('--optim_end', type=int, default=0, help='num of molecules to generate on each call to train.generate')

    parser.add_argument('--optimized_save_path', type=str, default='./anonymous/', help='path to save pos neg mean')

    parser.add_argument('--modify_size', type=int, default=5, help='num of molecules to generate on each call to train.generate')
    parser.add_argument('--repeat_time', type=int, default=100, help='num of molecules to generate on each call to train.generate')



    parser.add_argument('--reinforce_iters', type=int, default=200, help='number of iters for reinforce')
    parser.add_argument('--update_iters', type=int, default=4, help='number of iters for reinforce')

    parser.add_argument('--max_size_rl', type=int, default=38, help='maximal #atoms of generated molecule')
    parser.add_argument('--property', type=str, default='plogp', help='molecule property to optimize')

    parser.add_argument('--reward_decay', type=float, default=0.90, help='maximal #atoms of generated molecule')
    parser.add_argument('--reward_type', type=str, default='exp', help='maximal #atoms of generated molecule')

    parser.add_argument('--qed_coeff', type=float, default=1.0, help='maximal #atoms of generated molecule')
    parser.add_argument('--plogp_coeff', type=float, default=1.0, help='maximal #atoms of generated molecule')

    parser.add_argument('--exp_temperature', type=float, default=3.0, help='maximal #atoms of generated molecule')
    parser.add_argument('--exp_bias', type=float, default=4.0, help='maximal #atoms of generated molecule')

    parser.add_argument('--rl_sample_temperature', type=float, default=0.75, help='maximal #atoms of generated molecule')
    parser.add_argument('--moving_coeff', type=float, default=0.95, help='moving baseline coeff')
    parser.add_argument('--optim', type=str, default='adam', help='maximal #atoms of generated molecule')
    parser.add_argument('--penalty', action='store_false', default=True, help='reinforce')
    parser.add_argument('--lr_decay', action='store_true', default=False, help='reinforce')    
    parser.add_argument('--warm_up', type=int, default=0, help='linearly learning rate warmup')
    
    parser.add_argument('--split_batch', action='store_true', default=False, help='split the batch to two halves')

    parser.add_argument('--not_save_demon', action='store_true', default=False, help='reinforce')
    parser.add_argument('--take_min_act', type=int, default=0, help='take min action')
    parser.add_argument('--no_baseline', action='store_true', default=False, help='reinforce')

    parser.add_argument('--reinforce_fintune', action='store_true', default=False, help='reinforce')
    parser.add_argument('--optimize_mols_ckpt', action='store_true', default=False, help='reinforce')







    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.save:

        checkpoint_dir = args.all_save_prefix + 'reinforce_co_rl_rd%s_rltemp%s_%s_%s_seed%d_bsz%d_lr%f_wm%d_exp%d_msize%d' % (str(args.reward_decay), 
                                str(args.rl_sample_temperature), args.dataset, 
                                args.name, args.seed, 
                                args.batch_size, args.lr, args.warm_up, args.exp_temperature, args.modify_size)
        args.save_path = checkpoint_dir

        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
    set_seed(args.seed, args.cuda)

    print(args)

    assert args.co_rl and not args.train, 'please specify constrained optimization mode'
    assert (args.reinforce_fintune and not args.optimize_mols_ckpt) or (not args.reinforce_fintune and args.optimize_mols_ckpt)
  

    node_features, adj_features, mol_sizes, data_config, all_smiles, all_logps = read_smi_plogp(args.path)

    if args.reinforce_fintune:
        cur_dataloader = DataLoader(ConstrainOptim_Zink800(node_features, adj_features, mol_sizes, all_smiles, all_logps),
                                    batch_size=args.batch_size,
                                    shuffle=args.shuffle,
                                    num_workers=args.num_workers)
    elif args.optimize_mols_ckpt:
        cur_dataloader = ConstrainOptim_Zink800(node_features, adj_features, mol_sizes, all_smiles, all_logps)

       


    trainer = Trainer(cur_dataloader, data_config, args, all_train_smiles=all_smiles)
    if args.init_checkpoint is not None:
        trainer.initialize_from_checkpoint(gen=args.gen)
  
    if args.save:
        mol_out_dir = os.path.join(checkpoint_dir, 'mols')
        if not os.path.exists(mol_out_dir):
            os.makedirs(mol_out_dir)
    else:
        mol_out_dir = None  
    if args.reinforce_fintune:
        trainer.reinforce(mol_out_dir=mol_out_dir)
    elif args.optimize_mols_ckpt:
        trainer.constrained_optimize()
    #trainer.constrained_optimize(direction)
    

