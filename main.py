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
from utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

# Start Ray.
ray.init()
 
argparser = argparse.ArgumentParser()
argparser.add_argument('--game', '-g', default='braid', choices=['braid']) 
argparser.add_argument('--max_garside_len', default=100, type=int)
argparser.add_argument('--maxpad_of_product_matrix', default=315, type=int)
argparser.add_argument('--bias_canonical_form', default=-1.0, type=float)
argparser.add_argument('--numIters', default=10, type=int)
argparser.add_argument('--numEps', default=500, type=int)
argparser.add_argument('--tempThreshold', default=100, type=int)
argparser.add_argument('--updateThreshold', default=0.6, type=float)
argparser.add_argument('--maxlenOfQueue', default=200000, type=int)
argparser.add_argument('--numMCTSSims', default=25, type=int)
argparser.add_argument('--arenaCompare', default=40, type=int)
argparser.add_argument('--cpuct', default=1.1, type=float)
argparser.add_argument('--batch_size', default=128, type=int)
argparser.add_argument('--epochs', default=1, type=int)
argparser.add_argument('--checkpoint', default='./temp/')
argparser.add_argument('--load_model', default=False, type=bool)
argparser.add_argument('--load_folder_file', default=('/dev/models/8x100x50','best.pth.tar'))
argparser.add_argument('--numItersForTrainExamplesHistory', default=20, type=int)
argparser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool)
argparser.add_argument('--dropout', default=0.0, type=float)
argparser.add_argument('--mod_p', default=3, type=int)

args = argparser.parse_args()
EPS = 1e-8

# ./.venv/bin/python main.py --numEps 10
@ray.remote(num_cpus = 0.1, num_gpus=0.1 if args.cuda == True else 0, concurrency_groups={"execute": 16})
class Self_play:
    def __init__(self, policy, game, nnet, args):
        self.policy = policy
        self.game = game
        self.nnet = nnet
        self.args = args 
        self.mcts = MCTS(self.game, self.nnet, self.args)

    @ray.method(concurrency_group="execute")
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
        """
        
        trainExamples = []
        board = self.game.getInitBoard()
        curPlayer = 1
        episodeStep = 0

        last_action = 0
        projlens = [1]
        action_list = [0]
        for cur_garside_len in range(self.game.getBoardSize() + 1):
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, curPlayer)
            temp = int(episodeStep < args.tempThreshold)
        
            pi = self.mcts.getActionProb(canonicalBoard, cur_garside_len, last_action, temp=temp)
            sym = self.game.getSymmetries(canonicalBoard, pi)
            for b, p in sym:
                trainExamples.append([b, p, None])
            action = np.random.choice(len(pi), p=pi)
                
            if self.game.getGameEnded(cur_garside_len):
                best_projlen = np.inf
                # reward is minimum projlen attained after applying the action
                for i in range(len(trainExamples) - 2, -1, -1): 
                    x = trainExamples[i] 
                    if projlens[i] < best_projlen:
                        best_projlen = projlens[i]
                    trainExamples[i] = (x[0], x[1], self.game.transform_normalize_reward(best_projlen)) 
                return trainExamples[:-1], action_list, projlens
            
            try:
                board, next_garside_len, projlen = self.game.getNextState(board, action, cur_garside_len)
                projlens.append(projlen)
                last_action = action 
                action_list.append(action)
                if next_garside_len % 20 == 0: print ("pi", next_garside_len, pi, "action_list", action_list, "projlen", projlen)
            except:
                best_projlen = np.inf
                for i in range(len(trainExamples) - 1, -1, -1):
                    x = trainExamples[i] 
                    if projlens[i] < best_projlen:
                        best_projlen = projlens[i]
                    trainExamples[i] = (x[0], x[1], self.game.transform_normalize_reward(best_projlen)) 
                return trainExamples[:-1], action_list, projlens
            
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
        print ("mask_lookup_table", mask_lookup_table , len(mask_lookup_table) )
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
            return self.getCanonicalForm(board, None), cur_garside_len + 1, 1e3

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
        return -np.sign(projlen) * (np.log(np.abs(projlen) + 1)) + 4.5
    
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
        self.resnet = ResNet() # default: Resnet 18

        embed_size = 1000
        self.policy = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.BatchNorm1d(embed_size),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(embed_size, self.action_size)
        )

        self.value = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.BatchNorm1d(embed_size),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(embed_size, 1)
        )

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
        pi = self.policy(s)                                          # batch_size x action_size
        v = self.value(s)                                            # batch_size x 1
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

        if args.cuda:
            self.nnet.cuda()

    def train(self, examples, policy):
        """
        examples: list of examples, each example is of form (board, pi, v)

        if policy is random, only do value training
        """
        optimizer = torch.optim.Adam(self.nnet.parameters())

        for epoch in range(args.epochs):
            print('EPOCH ::: ' + str(epoch + 1))
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            batch_count = int(len(examples) / args.batch_size)
            print ("len(examples)", len(examples))
            t = tqdm(range(batch_count), desc='Training Net')
            for _ in t:
                optimizer.zero_grad()

                sample_ids = np.random.randint(len(examples), size=args.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # predict
                if args.cuda:
                    boards, target_pis, target_vs = boards.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()

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

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
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
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s

    def getActionProb(self, canonicalBoard, cur_garside_len, last_action, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Params:
            canonicalBoard: current board
            cur_garside_len: length of the braid before applying action
            last_action: last action taken by current player 
                (needed to mask out invalid moves)
            temp: temp=erature

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for i in range(self.args.numMCTSSims):
            self.search(canonicalBoard, cur_garside_len, last_action)

        s = self.game.stringRepresentation(canonicalBoard)
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

    def search(self, canonicalBoard, cur_garside_len, last_action):
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
            last_action: last action taken by current player 
                (needed to mask out invalid moves) 
        Returns:
            v: the negative of the value of the current canonicalBoard
        """

        s = self.game.stringRepresentation(canonicalBoard)
        
        if self.game.getGameEnded(cur_garside_len) != 0:
            # terminal node 
            print ("terminal node", s)
            if s not in self.Es:
                _, v = self.nnet.predict(canonicalBoard)
                self.Es[s] = v

            return self.Es[s]

        if s not in self.Ps:
            # leaf node 
            self.Ps[s], self.Es[s] = self.nnet.predict(canonicalBoard)
            v = self.Es[s] # value of the leaf node
            valids = self.game.getValidMoves(last_action) 
            # print ("valids", valids)
            self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
            # print ("masked self.Ps[s], v", s )
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                raise ValueError(f'All valid moves were masked. {self.Ps[s]}')
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                # log.error("All valid moves were masked, doing a workaround.")
                # self.Ps[s] = self.Ps[s] + valids
                # self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            return v


        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.game.getActionSize()):
             
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                            1 + self.Nsa[(s, a)])
                else:
                    u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act 
        # print ("best_act", s, best_act)
        next_s, next_cur_garside_len, _ = self.game.getNextState(canonicalBoard, a, cur_garside_len)
        next_s = self.game.getCanonicalForm(next_s, next_cur_garside_len)
        print ("recursion", cur_garside_len, s, self.game.stringRepresentation(next_s))
        v = self.search(next_s, next_cur_garside_len, a) # Search recursively

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1 
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
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples() 
        self.best_projlen_and_seq = {i: [None, np.inf] for i in range(self.args.max_garside_len + 1)}
    
    def executeEpisode(self, policy="net"):
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
        """
        trainExamples = []
        board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0

        last_action = 0
        projlens = [1]
        action_list = [0]
        for cur_garside_len in range(self.game.getBoardSize() + 1):
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)
            temp = int(episodeStep < self.args.tempThreshold)
            if policy == "net":
                pi = self.mcts.getActionProb(canonicalBoard, cur_garside_len, last_action, temp=temp)
                sym = self.game.getSymmetries(canonicalBoard, pi)
                for b, p in sym:
                    trainExamples.append([b, p, None])
                action = np.random.choice(len(pi), p=pi)
                
            elif policy == "random":
                pi = cur_garside_len
                valids = self.game.getValidMoves(last_action)
                valids = valids / np.sum(valids)  # make valids into probability dist
                action = np.random.choice(self.game.getActionSize(), p=valids)
                sym = self.game.getSymmetries(canonicalBoard, valids)
                for b, p in sym:
                    trainExamples.append([b, p, None])

            if self.game.getGameEnded(cur_garside_len):
                best_projlen = np.inf
                # reward is minimum projlen attained after applying the action
                for i in range(len(trainExamples) - 2, -1, -1): # assign values to each example
                    x = trainExamples[i] 
                    if projlens[i] < best_projlen:
                        best_projlen = projlens[i]
                    trainExamples[i] = (x[0], x[1], self.transform_normalize_reward(best_projlen))
                # print ("trainExamples", [i[0][0] for i in trainExamples])
                return trainExamples[:-1]
                
            try:
                board, next_garside_len, projlen = self.game.getNextState(board, action, cur_garside_len)
                projlens.append(projlen)
                last_action = action 
                action_list.append(action)
                print ("pi", pi, "action_list", action_list, "projlen", projlen)
                if projlen < self.best_projlen_and_seq[next_garside_len][1]:
                    self.best_projlen_and_seq[next_garside_len] = [action_list, projlen]  
            except: 
                best_projlen = np.inf
                for i in range(len(trainExamples) - 1, -1, -1): # assign values to each example
                    x = trainExamples[i] 
                    if projlens[i] < best_projlen:
                        best_projlen = projlens[i]
                    trainExamples[i] = (x[0], x[1], self.transform_normalize_reward(best_projlen))
                # print ("trainExamples", [i[2] for i in trainExamples])
                return trainExamples[:-1]
                
    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.args.numIters + 1):
            # bookkeeping
            log.info(f'Starting Iter #{i} ...')
            # examples of the iteration
            
            if i == 1: # Random play for the first iteration
                self.loadTrainExamples(i - 1)
                policy = "random"
                continue 
            elif i > 1: # NNet policy for the rest of the iterations
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)
                policy = "net"

                # self.mcts = MCTS(self.game, self.nnet, self.args)  # reset search tree
                # for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                #     iterationTrainExamples += self.executeEpisode(policy=policy)
                #     print ("best projlen", self.best_projlen_and_seq)
                # save the iteration examples to the history 
                 
                actors = [Self_play.remote(policy, self.game, self.nnet, self.args) for _ in range(self.args.numEps)]
                iterationTrainExamples = ray.get([actor.executeEpisode.remote() for actor in actors])
                print ("iterationTrainExamples", iterationTrainExamples)
                self.trainExamplesHistory.append(iterationTrainExamples)

                if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                    log.warning(
                        f"Removing the oldest entries in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                    self.trainExamplesHistory = self.trainExamplesHistory[-self.args.numItersForTrainExamplesHistory:]
                    
                # backup history to a file
                # NB! the examples were collected using the model from the previous iteration, so (i-1)  
                self.saveTrainExamples(i - 1)

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            print ("trainExamples", len(trainExamples))
            np.random.shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            # self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            # pmcts = MCTS(self.game, self.pnet, self.args)

            self.nnet.train(trainExamples, policy=policy)
            # nmcts = MCTS(self.game, self.nnet, self.args)

            # log.info('PITTING AGAINST PREVIOUS VERSION')
            # arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
            #               lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game)
            # pwins, nwins, draws = arena.playGames(self.args.arenaCompare)

            # log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            # if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.updateThreshold:
            #     log.info('REJECTING NEW MODEL')
            #     self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            # else:
            #     log.info('ACCEPTING NEW MODEL')
            #     self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
            #     self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + f"_{time.time()}.examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        fs = glob.glob(os.path.join(folder, self.getCheckpointFile(iteration) + "*.examples"))

        self.trainExamplesHistory = []
        for filename in fs:
            with open(filename, "rb") as f:
                self.trainExamplesHistory.extend(Unpickler(f).load())
        log.info('Loading done!')

        # examples based on the model were already collected (loaded)
        self.skipFirstSelfPlay = True


def main():
    log.info('Loading %s...', BraidGame.__name__)
    g = BraidGame(args.max_garside_len, args.maxpad_of_product_matrix)

    # log.info('Loading %s...', nn.__name__)
    nnet = NNetWrapper(g)

    # if args.load_model:
    #     log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
    #     nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    # else:
    #     log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    # if args.load_model:
    #     log.info("Loading 'trainExamples' from file...")
    #     c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    main()
