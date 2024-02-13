#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import time
import os
import logging
from collections import deque
import coloredlogs
from tqdm import tqdm
import math
import numpy as np
import operator
import functools
import copy
from typing import Sequence
from pickle import Pickler, Unpickler
import hashlib
import sys
import glob
import ray
import argparse

from peyl.braid import PermTable
from peyl import polymat, JonesCellRep, BraidGroup, GNF, Permutation
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

# from Coach import Coach
# from othello.OthelloGame import OthelloGame as Game
# from othello.pytorch.NNet import NNetWrapper as nn
import NeuralNet
import utils 

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

# Start Ray.
ray.init(ignore_reinit_error=True)
 
argparser = argparse.ArgumentParser()
argparser.add_argument('--game', '-g', default='braid', choices=['braid']) 
argparser.add_argument('--max_garside_len', default=100, type=int)
argparser.add_argument('--maxpad_of_product_matrix', default=315, type=int)
argparser.add_argument('--bias_symlog', default=4.5, type=float)
argparser.add_argument('--playout_cap_randomization_prob', default=0.25, type=float)
argparser.add_argument('--do_pretrain', action=argparse.BooleanOptionalAction, default=False)
argparser.add_argument('--startIter', default=1, type=int)
argparser.add_argument('--numIters', default=100, type=int)
argparser.add_argument('--numEps', default=100, type=int)
argparser.add_argument('--tempThreshold', default=200, type=int)
argparser.add_argument('--maxlenOfQueue', default=200000, type=int)
argparser.add_argument('--num_jobs_at_a_time', default=5, type=int)
argparser.add_argument('--nummaxMCTSSims', default=10, type=int)
argparser.add_argument('--numminMCTSSims', default=10, type=int)
argparser.add_argument('--cpuct', default=1.0, type=float)
argparser.add_argument('--batch_size', default=256, type=int)
argparser.add_argument('--pretrain_epochs', default=1, type=int)
argparser.add_argument('--epochs', default=10, type=int)
argparser.add_argument('--pretrain_lr', default=2e-5, type=float)
argparser.add_argument('--lr', default=6e-5, type=float)
argparser.add_argument('--checkpoint', default='./temp/')
argparser.add_argument('--load_model', default=False, type=bool)
argparser.add_argument('--load_checkpoint', default="", type=str)
argparser.add_argument('--numItersForTrainExamplesHistory', default=1000, type=int)
argparser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool)
argparser.add_argument('--dropout', default=0.0, type=float)
argparser.add_argument('--mod_p', default=3, type=int)
argparser.add_argument('--debug', action=argparse.BooleanOptionalAction, default=False)

if utils.is_interactive():
    jupyter_args = "--numEps 1 --num_jobs_at_a_time 1 --do_pretrain --startIter 11" \
            + " --epochs 1" 
    args = argparser.parse_args(args=jupyter_args.split())
else:
    args = argparser.parse_args()
print("configs", args, flush=True)
EPS = 1e-8

# RAY_DEDUP_LOGS=0 ./.venv/bin/python main.py --numEps 10
@ray.remote(num_cpus=0.2, num_gpus=0.2 if args.cuda == True else 0)
class Self_play:
    def __init__(self, policy, game, nnet, nummaxMCTSSims, numminMCTSSims, args):
        self.policy = policy
        self.game = game
        self.nnet = nnet
        self.nnet.set_eval_mode()
        self.nummaxMCTSSims = nummaxMCTSSims
        self.numminMCTSSims = numminMCTSSims
        self.args = args 
        self.mcts = MCTS(self.game, self.nnet, self.args)

    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                        pi is the MCTS informed policy vector, v is +1 if
                        the player eventually won the game, else -1.

        0 []
        1 [1, 4, 18]
        2 [2, 3, 12, 13, 16]
        3 [1, 4, 18]
        4 [2, 3, 12, 13, 16]
        5 [1, 2, 3, 4, 5, 12, 13, 16, 18, 19, 22]
        6 [6, 8, 9]
        7 [1, 4, 6, 7, 8, 9, 10, 11, 18, 20, 21]
        8 [2, 3, 12, 13, 16]
        9 [1, 4, 18]
        10 [2, 3, 12, 13, 16]
        11 [1, 2, 3, 4, 5, 12, 13, 16, 18, 19, 22]
        12 [6, 8, 9]
        13 [1, 4, 6, 7, 8, 9, 10, 11, 18, 20, 21]
        14 [2, 3, 6, 8, 9, 12, 13, 14, 15, 16, 17]
        15 [1, 4, 6, 7, 8, 9, 10, 11, 18, 20, 21]
        16 [2, 3, 12, 13, 16]
        17 [1, 2, 3, 4, 5, 12, 13, 16, 18, 19, 22]
        18 [6, 8, 9]
        19 [1, 4, 6, 7, 8, 9, 10, 11, 18, 20, 21]
        20 [2, 3, 6, 8, 9, 12, 13, 14, 15, 16, 17]
        21 [1, 4, 6, 7, 8, 9, 10, 11, 18, 20, 21]
        22 [2, 3, 6, 8, 9, 12, 13, 14, 15, 16, 17]
        23 [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
        """
        
        
        board = self.game.getInitBoard()
        trainExamples = []
        curPlayer = 1
        episodeStep = 0

        last_action = 0
        projlens = [1]
        action_list = (0,)
        for cur_garside_len in range(self.game.getBoardSize() + 1):
            # halt if game has ended or if projlen is 1 and cur_garside_len > 0
            # i.e. found kernel element
            if self.game.getGameEnded(cur_garside_len) or (projlens[-1] == 1 and cur_garside_len > 0):
                trainExamples = self.return_trainExamples(trainExamples, projlens) 
                # print ("trainExamples", trainExamples, len(trainExamples), "projlens", len(projlens), "action_list", len(action_list))
                # pad projlens to length max_garside_len
                projlens += [projlens[-1]] * (self.game.getBoardSize() - len(projlens[1:]))
                
                return trainExamples, action_list[1:], projlens[1:]
            
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, curPlayer)
            temp = int(episodeStep < args.tempThreshold)
            playout_cap_randomization = np.random.rand() < args.playout_cap_randomization_prob
            
            if playout_cap_randomization :
                # On a small proportion p of
                # turns, we perform a full search, stopping when the tree reaches a cap of N nodes, and for all
                pi = self.mcts.getActionProb(canonicalBoard, cur_garside_len, action_list, 
                                             num_playouts = self.nummaxMCTSSims, temp=temp,
                                             apply_Dirichlet_noise = True)
                sym = self.game.getSymmetries(canonicalBoard, pi)
                for b, p in sym:
                    trainExamples.append([b, p, cur_garside_len+1])
                action = np.random.choice(len(pi), p=pi)
            else:
                # other turns we perform a fast search with a much smaller cap of n < N. Only turns with a
                # full search are recorded for training. 
                pi = self.mcts.getActionProb(canonicalBoard, cur_garside_len, action_list, 
                                            num_playouts = self.numminMCTSSims, temp=temp,
                                            apply_Dirichlet_noise = False)
                action = np.random.choice(len(pi), p=pi)
            
            try:
                board, next_garside_len, projlen = self.game.getNextState(board, action, cur_garside_len)
                projlens.append(projlen)
                last_action = action 
                action_list += (action,)
                if next_garside_len % 20 == 0: print ("pi", next_garside_len, pi, "action_list", action_list, "projlen", projlen, flush=True)
            except:
                trainExamples = self.return_trainExamples(trainExamples, projlens)
                return trainExamples, action_list[1:], projlens[1:]

    def return_trainExamples(self, trainExamples, projlens):
        """
        This function adds the projlens to the trainExamples
        returns trainExamples, which is a dictionary of the form
        {projlen: [board, pi, projlen]}
        """
        trainExamples_dict = {}
        for i in range(len(trainExamples)):
            x = trainExamples[i] 
            trainExamples_dict[x[2]] = [x[0], x[1], projlens[x[2]]]
        return trainExamples_dict
    
class BraidGame():
    """
    This class specifies the base Game class. To define your own game, subclass
    this class and implement the functions below. This works when the game is
    two-player, adversarial and turn-based.

    Use 1 for player1 and -1 for player2.

    See othello/OthelloGame.py for an example implementation.
    """
    def __init__(self, max_garside_len, maxpad_of_product_matrix):
        """
        
        """
        self.max_garside_len = max_garside_len
        self.maxpad_of_product_matrix = maxpad_of_product_matrix
        nft = PermTable.create(n=4)
        cell_rep = JonesCellRep(n=4, r=1, p=args.mod_p)
        BG = BraidGroup(4)


        mask_lookup_table = copy.deepcopy(nft.follows)
        print ("mask_lookup_table")
        for i in range(24):
            print (i, mask_lookup_table[i])
        
        mask_lookup_table[0] =  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22] # mask for None token, 0
        masks = np.zeros((24, 24)) 
        for i in range(24):
            masks[i, mask_lookup_table[i]] = 1
            print (i, masks[i])
        self.masks = masks

        gen_lookup_table = {} 
        for i, gen in enumerate(nft.divs):
            gen_lookup_table[i] = polymat.from_matrix(cell_rep.evaluate(self.perms_to_braid(BG, [gen])), proj=True)
            gen_lookup_table[i] = polymat.trim(gen_lookup_table[i])  
        self.gen_lookup_table = gen_lookup_table

    def perms_to_braid(self, BG: BraidGroup, perms: Sequence[Permutation]) -> GNF:
        return functools.reduce(operator.mul, [BG.positive_lift(perm) for perm in perms], BG.id())


    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        # return np.zeros(self.max_garside_len, dtype = np.int32)
        return polymat.eye(3)

    def getBoardSize(self):
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        return self.max_garside_len

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
            allowable tokens are 0 to 22 (well 0 is only allowed to start sequences)
            and delta is not allowed 
            but we can just return 24
        """
        return 24 
        

    def getNextState(self, board, action, cur_garside_len):
        """
        Input:
            board: current board
            action: action taken by current player
            cur_garside_len: length of the braid before applying action

        Returns:
            nextBoard: board after applying action
            next_garside_len: length of the braid after applying action
        """
        try:
            board = copy.deepcopy(board)
            product_matrix_so_far = polymat.trim(board)
            product_matrix_so_far = polymat.trim(polymat.mul(product_matrix_so_far, self.gen_lookup_table[action])) 
            product_matrix_so_far = np.mod(product_matrix_so_far, args.mod_p)
            # print ("product_matrix_so_far", polymat.trim(board), product_matrix_so_far, product_matrix_so_far.shape, polymat.trim(product_matrix_so_far))
            return self.getCanonicalForm(product_matrix_so_far, None), cur_garside_len + 1, polymat.projlen(product_matrix_so_far)
        except:
            return self.getCanonicalForm(board, None), cur_garside_len + 1, polymat.projlen(board) + 4

    def transform_normalize_reward(self, projlen):
        """
        Use symlog transformation to normalize reward as in 
        https://arxiv.org/pdf/2301.04104.pdf along with bias
        so that projlen=100 is mapped to ~0.0

        negative projlen since we want to minimize projlen

        Input:
            projlen: projection length of the braid after applying action

        Returns:
            reward: reward for the action
        """ 
        return -np.sign(projlen) * (np.log(np.abs(projlen) + 1)) + args.bias_symlog
    
    def getnextproductmatrix(self, product_matrix_so_far, action):
        """
        Input:
            product_matrix_so_far: product matrix of the braid before applying action
            action: action taken by current player

        Returns:
            nextproductmatrix: product matrix of the braid after applying action
        """
        
    def getValidMoves(self, last_action):
        """
        Input:
            last_action

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
            validMoves only depends on the current move, so we can just return the mask for the current move
        """
        return self.masks[last_action]

    def getGameEnded(self, cur_garside_len):
        """
        Input:
            cur_garside_len: length of the braid before applying action

        Returns:
            r: 0 if game has not ended. 1 if game ended (max garside len reached)
               
        """
        return 1 if cur_garside_len >= self.max_garside_len else 0

    def getCanonicalForm(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be independent of player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
        """
        return polymat.zeropad(board, self.maxpad_of_product_matrix) 

    def getSymmetries(self, board, pi):
        """
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
            no symmetries known
            # TO DO: ask Daniel
        """
        return [(board,pi)]

    def stringRepresentation(self, board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """ 
        b = board.astype(np.int32).data.tobytes()
        return hashlib.sha256(b).hexdigest()

class NNet(nn.Module):
    def __init__(self, game, args):
        # game params
        
        self.max_garside_len = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        super(NNet, self).__init__() 
        self.resnet = utils.ResNet() # default: Resnet 18

        embed_size = 1000
        self.policy = nn.ModuleList ([nn.Linear(embed_size, embed_size) for i in range(6)]
                                     + [nn.Linear(embed_size, self.action_size)])
        self.value = nn.ModuleList ([nn.Linear(embed_size, embed_size) for i in range(6)] 
                                    + [nn.Linear(embed_size, 1)])
                                     
        
    def forward(self, s):
        """
        Input:
            s: batch_size x board_x x board_y
        Returns:
            pi: batch_size x self.getActionSize()
            v: batch_size x 1
        """ 
        
        s = 2 * s / (args.mod_p - 1) - 1 # normalize coefs of [0, mod_p - 1] to [-1, 1]
        # print ("s", s) 
        if s.dim() == 3:
            s = s.unsqueeze(0)

        s = self.resnet(s)
        pi = s
        for pl in self.policy[:-1]:
            pi = F.relu(pl(pi)) + pi
        pi = self.policy[-1](pi) 

        v = s
        for val in self.value[:-1]:
            v = F.relu(val(s)) + s
        v = self.value[-1](v)
        
        # print ("pi, v", (F.log_softmax(pi, dim=1)) , v)
        return F.log_softmax(pi, dim=1), v
     
class NNetWrapper(NeuralNet.NeuralNet):
    def __init__(self, game):
        self.nnet = NNet(game, args)
        # print no. of parameters 
        pytorch_total_params = sum(p.numel() for p in self.nnet.parameters())
        print ("pytorch_total_params", pytorch_total_params) 
        self.max_garside_len = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.optimizer = torch.optim.Adam(self.nnet.parameters(), 
                                    lr=0.1,
                                    weight_decay=1e-8)
        if args.cuda:
            self.nnet.cuda()

    def set_eval_mode(self):
        self.nnet.eval()

    def train(self, examples, policy, lr, num_epochs):
        """
        examples: list of examples, each example is of form (board, pi, v)
        policy: "net" or "random"
        lr: learning rate
        num_epochs: number of epochs

        if policy is random, only do value training
        """
        # Convert examples to tensors
        print ("examples", len(examples), len(examples[0]))
        boards, pis, vs, _ = list(zip(*examples))
        boards = torch.tensor(np.stack(boards))
        pis = torch.tensor(np.stack(pis))
        vs = torch.tensor(np.stack(vs).astype(np.float64)) 
        # drop nans from vs 
        # Create a dataset from tensors
        dataset = torch.utils.data.TensorDataset(boards, pis, vs)

        # Create a data loader
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        optimizer = self.optimizer
        # modify lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        for epoch in range(num_epochs):
            print('EPOCH ::: ' + str(epoch + 1))
            self.nnet.train()
            pi_losses = utils.AverageMeter()
            v_losses = utils.AverageMeter()
 
            t = tqdm(train_loader, desc='Training Net')

            for boards, target_pis, target_vs in t:

            
                optimizer.zero_grad()
                # predict
                if args.cuda:
                    boards, target_pis, target_vs = boards.cuda(), target_pis.cuda(), target_vs.cuda()

                # compute output
                out_pi, out_v = self.nnet(boards)

                l_pi = self.loss_pi(target_pis, out_pi)    
                l_v = self.loss_v(target_vs, out_v)

                if policy == "random":
                    total_loss = l_v
                else: 
                    total_loss = l_pi + l_v
                # record loss
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

                # compute gradient and do SGD step
                total_loss.backward()
                optimizer.step()

        return pi_losses.avg, v_losses.avg

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        start = time.time()

        # preparing input
        # board = torch.LongTensor(board.astype(np.int64))
        board = torch.FloatTensor(board.astype(np.float64)) 
        if args.cuda: board = board.contiguous().cuda()
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, filepath):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = None if args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])

class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args 
        # self.Qbias = {i: (-0.04247 *i+2.247) for i in range(args.max_garside_len)} # line from (0,2.247) to (100,-2), function of Qbias with cur_garside_len as parameter
        self.Qbias = {i: 0 for i in range(args.max_garside_len)} # remove Qbias but keep it in case we need it later
        # since we start off at "really good" states
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s

    def getActionProb(self, canonicalBoard, cur_garside_len, action_list, num_playouts, temp=1,
                      apply_Dirichlet_noise = True):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Params:
            canonicalBoard: current board
            cur_garside_len: length of the braid before applying action
            action_list: list of actions taken by current player
                we need last action taken by current player 
                (needed to mask out invalid moves)
            num_playouts: number of playouts
            temp: temp=erature
            apply_Dirichlet_noise: whether to apply Dirichlet noise to the root node

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for i in range(num_playouts):
            self.search(canonicalBoard, cur_garside_len, action_list, root=apply_Dirichlet_noise)

        s = self.game.stringRepresentation(canonicalBoard)
        s = action_list
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, canonicalBoard, cur_garside_len, action_list, root=False):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Params:
            canonicalBoard: current board
            cur_garside_len: length of the braid before applying action
            action_list: list of actions taken by current player
                we need last action taken by current player 
                (needed to mask out invalid moves)
        Returns:
            v: the negative of the value of the current canonicalBoard
        """

        # s = self.game.stringRepresentation(canonicalBoard)
        s = action_list
        last_action = action_list[-1] 
        print ("executing search on", s, "cur_garside_len", cur_garside_len, "root", root, flush=True)
        if self.game.getGameEnded(cur_garside_len) != 0:
            # terminal node 
            # print ("terminal node", s)
            if s not in self.Es:
                _, v = self.nnet.predict(canonicalBoard)
                self.Es[s] = v

            return self.Es[s]

        if s not in self.Ps:
            # leaf node 
            
            self.Ps[s], self.Es[s] = self.nnet.predict(canonicalBoard)
            # if : print ("Ps", self.Ps[s], "Es", self.Es[s])
            v = self.Es[s] # value of the leaf node
            valids = self.game.getValidMoves(last_action) 
            # print ("valids", valids)
            self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
            # print ("masked self.Ps[s], v", s )
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize 
            else:
                # raise ValueError(f'All valid moves were masked. {self.Ps[s]}')
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                
                p, v = self.nnet.nnet(torch.FloatTensor(canonicalBoard).cuda())
                log.error(f"All valid moves were masked, doing a workaround {p, v}")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 1
            print ("s", s, "not in Ps got v", v)
            return v

        # print ("s", s, "in Ps")        


        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1
        if root == True:
            num_valids = np.sum(valids) 
            noise = np.random.dirichlet([0.3] * int(num_valids)) 
            noise_id = 0
             
        
        prior_probability = {}

        # pick the action with the highest upper confidence bound
        us = []
        netp = {}
        for a in range(self.game.getActionSize()):
            if valids[a]:
                netp[a] = self.Ps[s][a]
                if root == True: # add dirichlet noise to the prior probability for the root node
                    
                    prior_probability[a] = 0.75 * self.Ps[s][a] + 0.25 * noise[noise_id]
                    noise_id += 1
                else:
                    prior_probability[a] = self.Ps[s][a]
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args.cpuct * prior_probability[a] * math.sqrt(self.Ns[s]) / (
                            1 + self.Nsa[(s, a)])
                    print ("u", s, "a", a, "u", u, "Q",self.Qsa[(s, a)], "netp", netp[a],  "prior_probability", prior_probability[a], "Ns", self.Ns[s], "UCB",self.args.cpuct * prior_probability[a] * math.sqrt(self.Ns[s]) / ( \
                            1 + self.Nsa[(s, a)]), flush=True)
                else:
                    u = self.Qbias[cur_garside_len] + self.args.cpuct * (prior_probability[a]) * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?
                    print ("u", s, "a", a, "u", u, "netp", netp[a],"prior_probability", prior_probability[a], "Ns", self.Ns[s], flush=True)
                us.append([a,u])
                if u > cur_best:
                    cur_best = u
                    best_act = a
                
        sort_us_by_u = sorted(us, key=lambda x: x[1], reverse=True)
        a = best_act 
        print ("best_act", s, best_act, sort_us_by_u)
        next_s, next_cur_garside_len, _ = self.game.getNextState(canonicalBoard, a, cur_garside_len)
        next_s = self.game.getCanonicalForm(next_s, next_cur_garside_len)
        # print ("recursion", cur_garside_len, s, self.game.stringRepresentation(next_s))
        # Search recursively
        v = self.search(next_s, next_cur_garside_len, action_list + (a,), root=False)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1 
        # print ("Ps", self.Ps.keys(), flush=True)
        return v

class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        # self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples() 
        self.best_projlen_and_seq = {i: [None, np.inf] for i in range(self.args.max_garside_len + 1)}
     
                
    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.

        Progressively increase the number of MCTS simulations used in the tree
        as training progresses.
        """
        if args.load_checkpoint != "":
            print("loading checkpoint", args.load_checkpoint)
            self.nnet.load_checkpoint(args.load_checkpoint)

        if self.args.do_pretrain:
            self.trainExamplesHistory = []
            for i in range(0, self.args.startIter):
                self.trainExamplesHistory.extend(self.loadTrainExamples(i))  

            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            print ("trainExamples", len(trainExamples)) 
            pi_loss, v_loss = self.nnet.train(trainExamples, policy="net", lr = self.args.pretrain_lr, num_epochs = self.args.pretrain_epochs)
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='pretrained.pth.tar')

        
        for i in range(self.args.startIter, self.args.numIters + 1):
            # bookkeeping
            log.info(f'Starting Iter #{i} ...')
            # examples of the iteration
             
            policy = "net"
            iterationTrainExamples = []
            nummaxMCTSSims = 10*(i - 12) + self.args.nummaxMCTSSims
            numminMCTSSims = 5*(i - 12) + self.args.numminMCTSSims
            for j in range(0, self.args.numEps, args.num_jobs_at_a_time):
                actors = [Self_play.remote(policy, self.game, self.nnet, nummaxMCTSSims, numminMCTSSims, self.args) for _ in range(args.num_jobs_at_a_time)]
                iterationTrainExamples += ray.get([actor.executeEpisode.remote() for actor in actors])

            iterationTrainExamples, action_lists, projlens = zip(*iterationTrainExamples) 
            # for game in iterationTrainExamples:
            #     for board, pi, proj_len in game:
            #         print (proj_len)
            # iterationTrainExamples: list of dict of the form cur_garside_len: (canonicalBoard, currPlayer, pi,v)
            
            # print ("iterationTrainExamples", len(iterationTrainExamples))
            # print ("action_lists", action_lists, flush=True)
            # print ("projlens", np.array(projlens).T, flush=True)
            projlens = np.array(projlens).T
            # print ("iterationTrainExamples", [game.keys() for game in iterationTrainExamples] , flush=True)

            projlen_across_batch = projlens[-1] # projlens[-1] is the projlen of the last action in the action_list
            min_max_normalized_projlen = (projlen_across_batch - projlen_across_batch.min()) / (projlen_across_batch.max() - projlen_across_batch.min() + 1e-8) # add 1e-8 to avoid division by zero
            min_max_normalized_projlen = -(2 * min_max_normalized_projlen - 1) # scale to [-1, 1] and negative projlen since we want to minimize projlen
            print ("min_max_normalized_projlen", projlen_across_batch, min_max_normalized_projlen, flush=True)
            for id, game in enumerate(iterationTrainExamples):
                for cur_garside_len in game:
                    _ = game[cur_garside_len][:]
                    game[cur_garside_len] = game[cur_garside_len][:2] + [min_max_normalized_projlen[id], projlens[cur_garside_len-1, id]]

                    print ("iterationTrainExamples[game][cur_garside_len]", _[2], game[cur_garside_len][2:], flush=True)
            iterationTrainExamples = [board_pi_v for game in iterationTrainExamples for _, board_pi_v in game.items()]

            
            # for (action_list, projlen) in zip(action_lists, projlens):
                 
            #     for l in range(1, len(action_list)):
            #         if projlen[l] < self.best_projlen_and_seq[l+1][1]:
            #             self.best_projlen_and_seq[l] = [action_list[:l], projlen[l]] 
            #         if projlen[l] == 1:
            #             print ("found kernel", action_list[:l])
            # print ("self.best_projlen_and_seq", self.best_projlen_and_seq)

            self.saveTrainExamples(iterationTrainExamples, i - 1)
            iterationTrainExamples = self.loadTrainExamples(i - 1)
            self.trainExamplesHistory.extend(iterationTrainExamples)
           
            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning(
                    f"Removing the oldest entries in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory = self.trainExamplesHistory[-self.args.numItersForTrainExamplesHistory:]
                 

            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            print ("trainExamples", self.trainExamplesHistory, len(trainExamples))

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            pi_loss, v_loss = self.nnet.train(trainExamples, policy=policy, lr = self.args.lr, num_epochs = self.args.epochs)

            self.logtrainingmetrics(action_lists, projlens, pi_loss, v_loss, i - 1)

    def getCheckpointFile(self, iteration):
        return 'testcheckpoint_epoch_' + str(iteration) + f'_mod_p_{args.mod_p}'

    def saveTrainExamples(self, iterationTrainExamples, iteration):
        if args.debug: return
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + f"_{time.time()}.examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(iterationTrainExamples)
        f.closed

    def logtrainingmetrics(self, action_lists, projlens, pi_loss, v_loss, iteration):
        if args.debug: return
        folder = self.args.checkpoint 
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, f"iteration_{iteration}_{time.time()}.actions_projlens") 
        action_lists_projlens = {
            "action_lists": action_lists,
            "projlens": projlens, 
            "pi_loss": pi_loss,
            "v_loss": v_loss,
        }
        with open(filename, "wb+") as f:
            Pickler(f).dump(action_lists_projlens)
        f.closed

    def loadTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        fs = glob.glob(os.path.join(folder, self.getCheckpointFile(iteration) + "*.examples"))

        iterationTrainExamples = []
        for filename in fs:
            with open(filename, "rb") as f:
                iterationTrainExamples.append(Unpickler(f).load())
        log.info(f'Loading done for iteration {iteration}!')
        return iterationTrainExamples


# In[ ]:


def main():
    log.info('Loading %s...', BraidGame.__name__)
    g = BraidGame(args.max_garside_len, args.maxpad_of_product_matrix)

    # log.info('Loading %s...', nn.__name__)
    nnet = NNetWrapper(g)


    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)
 
    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    main()

