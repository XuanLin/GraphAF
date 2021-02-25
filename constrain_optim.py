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

from model import GraphFlowModel
from dataloader import PretrainZinkDataset, ConstrainOptim_Zink800, PositiveNegativeZinkDataset
import environment as env

from rdkit import DataStructs
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import AllChem


def target_molecule_similarity(mol, target, radius=2, nBits=2048,
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
        print('restore from the latest checkpoint')
    else:
        restore_path = os.path.join(args.save_path, 'checkpoint%s' % str(epoch))
        print('restore from checkpoint%s' % str(epoch))

    checkpoint = torch.load(restore_path)
    model.load_state_dict(checkpoint['model_state_dict'])


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
        self.dataloader = dataloader
        self.data_config = data_config
        self.args = args
        self.all_train_smiles = all_train_smiles

        self.max_size = self.data_config['max_size']
        self.node_dim = self.data_config['node_dim'] - 1 # exclude padding dim.
        self.bond_dim = self.data_config['bond_dim']
       
        
        self._model = GraphFlowModel(self.max_size, self.node_dim, self.bond_dim, self.args.edge_unroll, self.args)
        self._optimizer = optim.Adam(filter(lambda p: p.requires_grad, self._model.parameters()),
                        lr=self.args.lr, weight_decay=self.args.weight_decay)

        self.best_loss = 100.0
        self.start_epoch = 0
        if self.args.cuda:
            self._model = self._model.cuda()
    

    def initialize_from_checkpoint(self, gen=False):
        checkpoint = torch.load(self.args.init_checkpoint)
        self._model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if not gen:
            self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.best_loss = checkpoint['best_loss']
            self.start_epoch = checkpoint['cur_epoch'] + 1
        print('initialize from %s Done!' % self.args.init_checkpoint)


    def constrained_optimize(self, direction):
        all_best_smile = []
        all_best_score = []
        #max_cnt = 1
        cur_cnt = 0
        data_len = len(self.dataloader)
        repeat_time = self.args.repeat_time
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
                batch_data = self.dataloader[batch_cnt]

                #inp_node_features = batch_data['node'] #(1, N, node_dim)
                inp_node_features = batch_data['node'].unsqueeze(0) #(1, N, node_dim)

                #inp_adj_features = batch_data['adj'] #(1, 4, N, N)              
                inp_adj_features = batch_data['adj'].unsqueeze(0) #(1, 4, N, N)              

                raw_smile = batch_data['raw_smile']  #(1)
                plogp = batch_data['plogp'] #(1)
                print(raw_smile, plogp)
                #print(inp_node_features)
                #print(inp_adj_features)
                best_smile0246, best_score0246, final_sim0246 = self.constrained_optimize_one_mol(direction, inp_adj_features, inp_node_features, plogp, raw_smile)
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

            final_cg_save_path = os.path.join(self.args.optimized_save_path, 's%dm%dr1%dr2%d' % (self.args.strength, 
                                                                self.args.modify_size, self.args.repeat_time, self.args.search_step))
            if not os.path.exists(final_cg_save_path):
                os.makedirs(final_cg_save_path)            
            save_one_optimized_molecule(final_cg_save_path, 
                                        raw_smile, plogp, best_smile_all, 
                                        best_score_all, cur_iter=batch_cnt, 
                                        ranges=ranges, sim=final_sim_all)
            cur_cnt += 1

            #print('sample: %d | repeat: %d | best smile: %s | best score: %s' % (batch_cnt, repeat_time, str(best_smile_all), str(best_score_all)))
        print('finish optimized %d molecules from %d to %d!' % (ranges[1] - ranges[0], ranges[0], ranges[1]))
        return 1

    def constrained_optimize_one_mol(self, direction, adj, x, org_plogp, org_smile):
        """
        direction: score ascent direction
        adj: adjacent matrix of origin mol (1, 4, N, N)
        x: node feature of origin mol (1, N, 9)
        """
        #get original latent
        #fp1 = AllChem.GetMorganFingerprint(mol1, 2)
        #fp2 = AllChem.GetMorganFingerprint(mol2, 2)
        #DataStructs.TanimotoSimilarity(fp1, fp2) 

        direction = direction / direction.norm() #l2 norm normalize
        self._model.eval()
        with torch.no_grad():
            if self.args.cuda:
                adj = adj.cuda()
                x = x.cuda()
                direction = direction.cuda()
            out_z, out_logdet, ln_var = self._model(x, adj)
            node_latent = out_z[0].view(-1) #(d)
            edge_latent = out_z[1].view(-1) #(d)
        all_latent = torch.cat((node_latent, edge_latent), dim=0) #(1854, )
        step = self.args.search_step
        step_length = 0.1
        all_smiles = []
        all_plogps = []
        best_smile0 = None
        best_smile2 = None
        best_smile4 = None
        best_smile6 = None
        best_score0 = -100.
        best_score2 = -100.
        best_score4 = -100.
        best_score6 = -100.
        final_sim0 = -1.
        final_sim2 = -1.
        final_sim4 = -1.
        final_sim6 = -1.


        mol_org = Chem.MolFromSmiles(org_smile)
        mol_org_size = mol_org.GetNumAtoms()
        #fp_org = AllChem.GetMorganFingerprint(mol_org, 2)
        print('****** %s ******' % org_smile)
        for cur_step in range(0, step):
            #if cur_step != 0:
            #    continue
            #cur_latent = all_latent + direction * step_length * cur_step
            cur_latent = all_latent + direction * self.args.strength * torch.rand(direction.size()).cuda()

            smiles, no_resample, num_atoms = self._model.generate_one_molecule_with_latent_provided(cur_latent, all_latent, mol_org_size-self.args.modify_size,
                                                                    max_atoms=self.args.max_atoms, temperature=self.args.temperature)
            all_smiles.append(smiles)
            mol = Chem.MolFromSmiles(smiles)
            plogp_score = env.penalized_logp(mol)            

            s_tmp1 = Chem.MolToSmiles(mol, isomericSmiles=True) # this is important, or the score is not consistent with the original molecule.
            s_tmp2 = Chem.MolToSmiles(mol, isomericSmiles=False) # this is important, or the score is not consistent with the original molecule.

            mol = Chem.MolFromSmiles(s_tmp1)
            plogp_score = min(env.penalized_logp(mol), plogp_score)
            mol = Chem.MolFromSmiles(s_tmp2)
            plogp_score = min(env.penalized_logp(mol), plogp_score)

            all_plogps.append(plogp_score)
            #fp_cur = AllChem.GetMorganFingerprint(mol, 2)
            sim = target_molecule_similarity(mol_org, mol)
            #sim = DataStructs.TanimotoSimilarity(fp_org, fp_cur) 
            print('step: %d | smiles: %s | sim: %.3f | original plogp: %.5f | cur plogp: %.5f' % (cur_step, smiles, sim, org_plogp, plogp_score))           
            if sim >= 0:
                if plogp_score > best_score0:
                    best_score0 = plogp_score
                    best_smile0 = smiles
                    final_sim0 = sim

            if sim >= 0.2:
                if plogp_score > best_score2:
                    best_score2 = plogp_score
                    best_smile2 = smiles
                    final_sim2 = sim
            if sim >= 0.4:
                if plogp_score > best_score4:
                    best_score4 = plogp_score
                    best_smile4 = smiles
                    final_sim4 = sim
            if sim >= 0.6:
                if plogp_score > best_score6:
                    best_score6 = plogp_score
                    best_smile6 = smiles                                        
                    final_sim6 = sim

        return [best_smile0, best_smile2, best_smile4, best_smile6], [best_score0, best_score2, best_score4, best_score6], [final_sim0, final_sim2, final_sim4, final_sim6]




    def generate_molecule(self, num=100, epoch=None, out_path=None, mute=False):
        self._model.eval()
        all_smiles = []
        pure_valids = []
        appear_in_train = 0.
        start_t = time()
        cnt_mol = 0
        cnt_gen = 0

        while cnt_mol < num:
            smiles, no_resample, num_atoms =  self._model.generate(self.args.temperature, mute=mute, max_atoms=self.args.max_atoms, cnt=cnt_gen)
            cnt_gen += 1

            if num_atoms < self.args.min_atoms:
                print('#atoms of generated molecule less than %d, discarded!' % self.args.min_atoms)
            else:
                cnt_mol += 1

                if cnt_mol % 100 == 0:
                    print('cur cnt mol: %d' % cnt_mol)                

                all_smiles.append(smiles)
                pure_valids.append(no_resample)
                if self.all_train_smiles is not None and smiles in self.all_train_smiles:
                    appear_in_train += 1.0

            mol = Chem.MolFromSmiles(smiles)
            qed_score = env.qed(mol)
            plogp_score = env.penalized_logp(mol)


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





    def fit(self, mol_out_dir=None):        
        t_total = time()
        total_loss = []
        best_loss = self.best_loss
        start_epoch = self.start_epoch
        all_unique_rate = []
        all_valid_rate = []
        all_novelty_rate = []
        print('start fitting.')
        for epoch in range(self.args.epochs):
            epoch_loss = self.train_epoch(epoch + start_epoch)
            total_loss.append(epoch_loss)

            mol_save_path = os.path.join(mol_out_dir, 'epoch%d.txt' % (epoch + start_epoch)) if mol_out_dir is not None else None
            cur_unique, cur_valid, cur_novelty = self.generate_molecule(num=100, epoch=epoch + start_epoch, 
                                        out_path=mol_save_path, mute=True)

            all_unique_rate.append(cur_unique)
            all_valid_rate.append(cur_valid)
            all_novelty_rate.append(cur_novelty)

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                if self.args.save:
                    var_list = {'cur_epoch': epoch + start_epoch,
                                'best_loss': best_loss,

                               }
                    save_model(self._model, self._optimizer, self.args, var_list, epoch=epoch + start_epoch)
        

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time() - t_total))
        if mol_out_dir is not None and self.args.save:
            all_unique_rate = np.array(all_unique_rate)
            all_valid_rate = np.array(all_valid_rate)
            all_novelty_rate = np.array(all_novelty_rate)
            print('saving unique and valid array...')
            np.save(os.path.join(mol_out_dir, 'unique'), all_unique_rate)
            np.save(os.path.join(mol_out_dir, 'valid'), all_valid_rate)
            np.save(os.path.join(mol_out_dir, 'novelty'), all_novelty_rate)               



    def train_epoch(self, epoch_cnt):
        t_start = time()
        batch_losses = []
        self._model.train()
        batch_cnt = 0
        epoch_example = 0
        for i_batch, batch_data in enumerate(self.dataloader):
            batch_time_s = time()

            self._optimizer.zero_grad()

            batch_cnt += 1
            inp_node_features = batch_data['node'] #(B, N, node_dim)
            inp_adj_features = batch_data['adj'] #(B, 4, N, N)            
            if self.args.cuda:
                inp_node_features = inp_node_features.cuda()
                inp_adj_features = inp_adj_features.cuda()
            if self.args.deq_type == 'random':
                out_z, out_logdet, ln_var = self._model(inp_node_features, inp_adj_features)

                loss = self._model.log_prob(out_z, out_logdet)


                #TODO: add mask for different molecule size, i.e. do not model the distribution over padding nodes.

            elif self.args.deq_type == 'variational':
                out_z, out_logdet, out_deq_logp, out_deq_logdet = self._model(inp_node_features, inp_adj_features)
                ll_node, ll_edge, ll_deq_node, ll_deq_edge = self._model.log_prob(out_z, out_logdet, out_deq_logp, out_deq_logdet)
                loss = -1. * ((ll_node-ll_deq_node) + (ll_edge-ll_deq_edge))
            else:
                raise ValueError('unsupported dequantization method: (%s)' % self.deq_type)

            loss.backward()
            self._optimizer.step()


            batch_losses.append(loss.item())

            if batch_cnt % self.args.show_loss_step == 0 or (epoch_cnt == 0 and batch_cnt <= 100):
                #print(out_z[0][0])
                epoch_example = [out_z[0][0], out_z[1][0]]
                print('epoch: %d | step: %d | time: %.5f | loss: %.5f | ln_var: %.5f' % (epoch_cnt, batch_cnt, time() - batch_time_s, batch_losses[-1], ln_var))

        epoch_loss = sum(batch_losses) / len(batch_losses)
        print(epoch_example)
        print('Epoch: {: d}, loss {:5.5f}, epoch time {:.5f}'.format(epoch_cnt, epoch_loss, time()-t_start))          
        return epoch_loss

        
    def get_pos_neg_mean(self):
        t_start = time()
        self._model.eval()
        batch_cnt = 0
        sample_cnt = 0
        node_latent = None
        edge_latent = None
        with torch.no_grad():
            for i_batch, batch_data in enumerate(self.dataloader):
                batch_cnt += 1
                inp_node_features = batch_data['node'] #(B, N, node_dim)
                inp_adj_features = batch_data['adj'] #(B, 4, N, N)
                if self.args.cuda:
                    inp_node_features = inp_node_features.cuda()
                    inp_adj_features = inp_adj_features.cuda()
                if self.args.deq_type == 'random':
                    out_z, out_logdet, ln_var = self._model(inp_node_features, inp_adj_features) # shape of out_z: (b,d)
                    sample_cnt += out_z[0].size(0)
                    assert out_z[0].size(0) == out_z[1].size(0)
                    if batch_cnt == 1:
                        node_latent = out_z[0].sum(0)
                        edge_latent = out_z[1].sum(0)
                    else:
                        node_latent += out_z[0].sum(0)
                        edge_latent += out_z[1].sum(0)
                else:
                    raise ValueError('unsupported dequantization type')
                if batch_cnt % 100 == 0:
                    print('cur batch: %d' % batch_cnt)

        node_latent_avg = node_latent.cpu() / sample_cnt
        edge_latent_avg = edge_latent.cpu() / sample_cnt
        return node_latent_avg, edge_latent_avg



    def get_pos_neg_latent(self):
        t_start = time()
        self._model.eval()
        batch_cnt = 0
        sample_cnt = 0
        node_latents = []
        edge_latents = []
        with torch.no_grad():
            for i_batch, batch_data in enumerate(self.dataloader):
                batch_cnt += 1
                inp_node_features = batch_data['node'] #(B, N, node_dim)
                inp_adj_features = batch_data['adj'] #(B, 4, N, N)
                if self.args.cuda:
                    inp_node_features = inp_node_features.cuda()
                    inp_adj_features = inp_adj_features.cuda()
                if self.args.deq_type == 'random':
                    out_z, out_logdet, ln_var = self._model(inp_node_features, inp_adj_features) # shape of out_z: (b,d)
                    sample_cnt += out_z[0].size(0)
                    assert out_z[0].size(0) == out_z[1].size(0)

                    node_latents.append(out_z[0].cpu())
                    edge_latents.append(out_z[1].cpu())
                else:
                    raise ValueError('unsupported dequantization type')
                if batch_cnt % 100 == 0:
                    print('cur batch: %d' % batch_cnt)
        node_latents = torch.cat(node_latents, dim=0)
        edge_latents = torch.cat(edge_latents, dim=0)
        print('node latents shape after cat1', node_latents.size()) #(batch, d1)
        print('edge latents shape after cat1', edge_latents.size()) # (batch, d2)

        final_latents = torch.cat([node_latents, edge_latents], dim=1) #(batch, d1+d2)
        print('final latents shape', final_latents.size())
        #node_latent_avg = node_latent.cpu() / sample_cnt
        #edge_latent_avg = edge_latent.cpu() / sample_cnt
        return final_latents






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

    parser.add_argument('--co', action='store_true', default=False, help='generate')
    parser.add_argument('--get_mean', action='store_true', default=False, help='generate')
    parser.add_argument('--get_visualize_latent', action='store_true', default=False, help='generate')
    
    parser.add_argument('--optimize_one_mol', action='store_true', default=False, help='generate')
    parser.add_argument('--optim_start', type=int, default=0, help='num of molecules to generate on each call to train.generate')
    parser.add_argument('--optim_end', type=int, default=0, help='num of molecules to generate on each call to train.generate')


    parser.add_argument('--pos_neg_mean_save_path', type=str, help='path to save pos neg mean')
    parser.add_argument('--pos_neg_latent_save_path', type=str, help='path to save pos neg mean')

    parser.add_argument('--optimized_save_path', type=str, default='./anonymous/', help='path to save pos neg mean')
    parser.add_argument('--strength', type=float, default=100., help='num of molecules to generate on each call to train.generate')
    parser.add_argument('--modify_size', type=int, default=10, help='num of molecules to generate on each call to train.generate')

    parser.add_argument('--split_batch', action='store_true', default=False, help='split the batch to two halves')
    parser.add_argument('--repeat_time', type=int, default=40, help='num of molecules to generate on each call to train.generate')
    parser.add_argument('--search_step', type=int, default=5, help='num of molecules to generate on each call to train.generate')








    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.save:
        checkpoint_dir = args.all_save_prefix + 'save_co/%s_%s_%s' % (args.st_type, args.dataset, args.name)
        args.save_path = checkpoint_dir

        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
    set_seed(args.seed, args.cuda)

    print(args)

    assert args.co and not args.train, 'please specify constrained optimization mode'
    assert (not args.get_mean and args.optimize_one_mol) or (not args.optimize_one_mol and args.get_mean) or (args.get_visualize_latent)

    if args.get_mean or args.get_visualize_latent:
        print('get pos neg mean')
        node_features, adj_features, mol_sizes, data_config, all_smiles = read_molecules(args.path)
        cur_dataloader = DataLoader(PositiveNegativeZinkDataset(node_features, adj_features, mol_sizes),
                                  batch_size=args.batch_size,
                                  shuffle=args.shuffle,
                                  num_workers=args.num_workers)

    elif args.optimize_one_mol:
        print('optimize one mol')
        node_features, adj_features, mol_sizes, data_config, all_smiles, all_logps = read_smi_plogp(args.path)

        pos_latent_avg = torch.load(args.all_save_prefix + 'co_mean/pos_half_mean399.latent')
        neg_latent_avg = torch.load(args.all_save_prefix + 'co_mean/neg_half_mean399.latent')
        direction = pos_latent_avg - neg_latent_avg

        #cur_dataloader = DataLoader(ConstrainOptim_Zink800(node_features, adj_features, mol_sizes, all_smiles, all_logps),
        #                          batch_size=1,
        #                          shuffle=False,
        #                          num_workers=args.num_workers)
        cur_dataloader = ConstrainOptim_Zink800(node_features, adj_features, mol_sizes, all_smiles, all_logps)
       


    trainer = Trainer(cur_dataloader, data_config, args, all_train_smiles=all_smiles)
    if args.init_checkpoint is not None:
        trainer.initialize_from_checkpoint(gen=args.gen)
    '''
    if args.train:
        if args.save:
            mol_out_dir = os.path.join(checkpoint_dir, 'mols')

            if not os.path.exists(mol_out_dir):
                os.makedirs(mol_out_dir)
        else:
            mol_out_dir = None
        trainer.fit(mol_out_dir=mol_out_dir)

    if args.gen:
        print('start generating...')
        trainer.generate_molecule(num=args.gen_num, out_path=args.gen_out_path, mute=False)
    '''
    if args.get_mean:
        node_latent_avg, edge_latent_avg = trainer.get_pos_neg_mean()
        print('node latent avg size', node_latent_avg.size())
        print('edge latent avg size', edge_latent_avg.size())
        final_mean = torch.cat((node_latent_avg, edge_latent_avg), dim=0)
        torch.save(final_mean, args.pos_neg_mean_save_path)
    elif args.get_visualize_latent:
        final_latents = trainer.get_pos_neg_latent()
        torch.save(final_latents, args.pos_neg_latent_save_path)
    if args.optimize_one_mol:
        trainer.constrained_optimize(direction)
    

