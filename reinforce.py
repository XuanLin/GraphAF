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


def read_molecules(path):
    print('reading data from %s' % path)
    node_features = np.load(path + '_node_features.npy')
    adj_features = np.load(path + '_adj_features.npy')
    mol_sizes = np.load(path + '_mol_sizes.npy')

    f = open(path + '_config.txt', 'r')
    data_config = eval(f.read())
    f.close()


    fp = open(path + '_raw_smiles.smi')
    all_smiles = []
    for smiles in fp:
        all_smiles.append(smiles.strip())
    fp.close()
    return node_features, adj_features, mol_sizes, data_config, all_smiles

    
class Trainer_RL(object):
    def __init__(self, data_config, args, all_train_smiles=None):
        self.data_config = data_config
        self.args = args
        self.all_train_smiles = all_train_smiles

        self.max_size = self.data_config['max_size']
        self.node_dim = self.data_config['node_dim'] - 1 # exclude padding dim.
        self.bond_dim = self.data_config['bond_dim']
       
        
        self._model = GraphFlowModel(self.max_size, self.node_dim, self.bond_dim, self.args.edge_unroll, self.args)
        if self.args.optim == 'adam':
            self._optimizer_rl = optim.Adam(filter(lambda p: p.requires_grad, self._model.flow_core.parameters()),
                        lr=self.args.lr, weight_decay=self.args.weight_decay)
        elif self.args.optim == 'rmsprop':
            self._optimizer_rl = optim.RMSprop(filter(lambda p: p.requires_grad, self._model.parameters()),
                        lr=self.args.lr, weight_decay=self.args.weight_decay)
        if self.args.lr_decay:
            self._scheduler = optim.lr_scheduler.ReduceLROnPlateau(self._optimizer_rl, 'max', patience=2, factor=0.5, min_lr=1e-6)
        self.best_reward = -100.0
        self.start_iter = 0
        if self.args.cuda:
            self._model = self._model.cuda()
        print(self._model.state_dict().keys()) # check the key consistency.           
    
    def initialize_from_checkpoint(self):
        #TODO: check ckpt consistency
        checkpoint = torch.load(self.args.init_checkpoint)
        print(checkpoint['model_state_dict'].keys()) # check the key consistency     
        self._model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        if 'reinforce' in self.args.init_checkpoint:
            print('loading optim state from reinforce checkpoint...')
            self._optimizer_rl.load_state_dict(checkpoint['optimizer_state_dict'])
            self.best_reward = checkpoint['best_reward']
            self.start_iter = checkpoint['cur_iter'] + 1

        print('initialize from %s Done!' % self.args.init_checkpoint)



    def generate_molecule(self, num=100, epoch=None, out_path=None, mute=False):
        self._model.eval()
        all_smiles = []
        pure_valids = []
        appear_in_train = 0.
        start_t = time()
        cnt_mol = 0
        cnt_gen = 0

        while cnt_mol < num:
            smiles, no_resample, num_atoms =  self._model.generate(self.args.temperature, mute=mute, max_atoms=self.args.max_atoms)
            cnt_gen += 1
            if num_atoms >= self.args.min_atoms: # we constrain the size of generated molecule larger than min_atoms
                cnt_mol += 1
                all_smiles.append(smiles)
                pure_valids.append(no_resample)
                if self.all_train_smiles is not None and smiles in self.all_train_smiles: # count novelty
                    appear_in_train += 1.0

        assert cnt_mol == num, 'number of generated molecules does not equal num'
        unique_smiles = list(set(all_smiles))
        unique_rate = len(unique_smiles) / num
        pure_valid_rate = sum(pure_valids) / num
        novelty = 1. - (appear_in_train / num)

        if epoch is None:
            print('Time of generating (%d/%d) molecules(#atoms>=%d): %.5f | unique rate: %.5f | valid rate: %.5f | novelty: %.5f' % (num, 
                                cnt_gen, self.args.min_atoms, time()-start_t, unique_rate, pure_valid_rate, novelty))
        else:
            print('Time of generating (%d/%d) molecules(#atoms>=%d): %.5f at epoch :%d | unique rate: %.5f | valid rate: %.5f | novelty: %.5f' % (num, 
                                cnt_gen, self.args.min_atoms, time()-start_t, epoch, unique_rate, pure_valid_rate, novelty))
        if out_path is not None and self.args.save:
            fp = open(out_path, 'w')
            cnt = 0
            for i in range(len(all_smiles)):
                fp.write(all_smiles[i] + '\n')
                cnt += 1
            fp.close()
            print('writing %d smiles into %s done!' % (cnt, out_path))
        return (unique_rate, pure_valid_rate, novelty)



    def reinforce(self, mol_out_dir=None):        
        t_total = time()
        total_loss = []
        total_reward = []
        total_score = []
        best_reward = self.best_reward
        start_iter = self.start_iter
        #moving_baseline = np.zeros([self.args.max_size_rl])
        moving_baseline = None
        print('start finetuning model(reinforce)')
        for cur_iter in range(self.args.reinforce_iters):
            if cur_iter == 0:
                iter_loss, iter_reward, iter_score, moving_baseline = self.reinforce_one_iter(cur_iter + start_iter, in_baseline=None)
            else:
                iter_loss, iter_reward, iter_score, moving_baseline = self.reinforce_one_iter(cur_iter + start_iter, in_baseline=moving_baseline)

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

        loss, per_mol_reward, per_mol_property_score, out_baseline = self._model.reinforce_forward(temperature=self.args.rl_sample_temperature, 
                                                max_size_rl=self.args.max_size_rl, batch_size=self.args.batch_size, in_baseline=in_baseline, cur_iter=iter_cnt)

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
        return loss.item(), avg_reward, avg_score, out_baseline



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GraphFlow model')

    # ******data args******
    parser.add_argument('--dataset', type=str, default='zinc250k', help='dataset')
    parser.add_argument('--path', type=str, help='path of dataset', required=True)

    parser.add_argument('--batch_size', type=int, default=32, help='batch_size.')
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
    parser.add_argument('--lr', type=float, default=1e-6, help='Initial learning rate.')
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
    parser.add_argument('--min_atoms', type=int, default=10, help='minimum #atoms of generated mol, otherwise the mol is simply discarded')
    parser.add_argument('--max_atoms', type=int, default=48, help='maximum #atoms of generated mol')    
    parser.add_argument('--gen_num', type=int, default=100, help='num of molecules to generate on each call to train.generate')
    parser.add_argument('--gen', action='store_true', default=False, help='generate')
    parser.add_argument('--gen_out_path', type=str, help='output path for generated mol')

    # ******reinfroce args******
    parser.add_argument('--all_save_prefix', type=str, default='./', help='path of save prefix')

    parser.add_argument('--reinforce', action='store_true', default=False, help='reinforce')
    parser.add_argument('--reinforce_iters', type=int, default=5000, help='number of iters for reinforce')
    parser.add_argument('--update_iters', type=int, default=4, help='number of iters for reinforce')

    parser.add_argument('--max_size_rl', type=int, default=48, help='maximal #atoms of generated molecule')
    parser.add_argument('--property', type=str, help='molecule property to optimize')

    parser.add_argument('--reward_decay', type=float, default=0.90, help='maximal #atoms of generated molecule')
    parser.add_argument('--reward_type', type=str, default='exp', help='maximal #atoms of generated molecule')

    parser.add_argument('--qed_coeff', type=float, default=1.0, help='maximal #atoms of generated molecule')
    parser.add_argument('--plogp_coeff', type=float, default=1/3, help='maximal #atoms of generated molecule')

    parser.add_argument('--exp_temperature', type=float, default=3.0, help='maximal #atoms of generated molecule')
    parser.add_argument('--exp_bias', type=float, default=4.0, help='maximal #atoms of generated molecule')

    parser.add_argument('--rl_sample_temperature', type=float, default=0.75, help='maximal #atoms of generated molecule')
    parser.add_argument('--moving_coeff', type=float, default=0.95, help='moving baseline coeff')
    parser.add_argument('--optim', type=str, default='adam', help='maximal #atoms of generated molecule')
    parser.add_argument('--penalty', action='store_false', default=True, help='reinforce')
    parser.add_argument('--lr_decay', action='store_true', default=False, help='reinforce')
    parser.add_argument('--warm_up', type=int, default=0, help='linearly learning rate warmup')
    
    parser.add_argument('--split_batch', action='store_true', default=False, help='split the batch to two halves')







    print(torch.rand(1))
    print(np.random.randint(10000000000))
    print(random.randint(0, 10000000000000))


    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.save:
        if args.property == 'qed':
            coeff = args.qed_coeff
        elif args.property == 'plogp':
            coeff = args.plogp_coeff
        else:
            raise ValueError('unsupported property type, choices are [qed, plogp]')

        checkpoint_dir = args.all_save_prefix + 'rl_save/reinforce_%scoeff%s_rd%s_rltemp%s_%s_%s_seed%d_bsz%d_lr%f_wm%d_exp%d' % (args.property, str(coeff), str(args.reward_decay), 
                                                                                                    str(args.rl_sample_temperature), args.dataset, 
                                                                                                    args.name, args.seed, 
                                                                                                    args.batch_size, args.lr, args.warm_up, args.exp_temperature)
        args.save_path = checkpoint_dir

        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)

    set_seed(args.seed, args.cuda)
    print(torch.rand(1))
    
    print(np.random.randint(10000000000))
    print(random.randint(0, 10000000000000))
    print(args)

    assert (not args.train) and (args.reinforce or args.gen), 'this file is only used for finetuning...'
    node_features, adj_features, mol_sizes, data_config, all_smiles = read_molecules(args.path)

    del node_features
    del adj_features
    del mol_sizes


    trainer = Trainer_RL(data_config, args, all_train_smiles=all_smiles)

    if args.init_checkpoint is not None:
        trainer.initialize_from_checkpoint()

    if args.save:
        mol_out_dir = os.path.join(checkpoint_dir, 'mols')
        if not os.path.exists(mol_out_dir):
            os.makedirs(mol_out_dir)
    else:
        mol_out_dir = None

    if args.reinforce:
        trainer.reinforce(mol_out_dir=mol_out_dir)

    if args.gen:
        print('start generating...')
        trainer.generate_molecule(num=args.gen_num, out_path=args.gen_out_path, mute=False)
