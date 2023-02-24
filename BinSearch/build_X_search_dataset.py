import os
os.environ['CUDA_VISIBLE_DEVICES']='0,1'
import torch.backends.cudnn as cudnn
import argparse
import torch
from torch import nn

import random
import time
import json
from datetime import datetime

from sklearn import metrics
import numpy as np
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from getSearchBase import pairsDataset
# path must be identical with training pass.
from Model.GraphLSTMWithKB import  graphLSTMCatNetwork


USE_CUDA = torch.cuda.is_available()

cudnn.benchmark = True

random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)

if USE_CUDA:
    DEVICE = torch.device('cuda' )
    print('***************')
    print('-----cuda------')
    print('***************')
    torch.cuda.manual_seed(1234)
else:
    DEVICE = torch.device('cpu')


class x2vTrainer(object):
    def __init__(self, flags):
        self.network = flags.network
        self.best_network_model = flags.model
        self.epochs = 10
        self.batch_size = 50

        self.max_instructions = 50
        self.max_nodes = 150

        # seq
        self.seq_len = 150
        # kb seq length for all basic block with cat( func= graph + kb)
        self.kb_seq = 150
        # graph for node kb length
        self.node_kb_max = 5

        # path
        self.max_path = 2
        self.path_len = 30

        # self.lstm_batch_size = self.batch_size * self.max_nodes
        self.lr = 0.001

        # i2v
        self.embedding_dim = 100
        self.hidden_dim = 100

        self.output_dim = 64
        self.num_layers = 1

        self.n_heads = 4

        # acfg
        # self.acfg_length = 8
        self.acfg_length = 7


        self.get_data = pairsDataset(flags.word2id, flags.api2id,flags.search_dataset,self.seq_len,
                                          self.max_instructions,
                                          self.max_nodes,
                                          self.kb_seq,
                                          self.node_kb_max,
                                          self.max_path,
                                          self.path_len
                                          )

        self.id2vec = self.matrix_load(flags.id2vec)
        self.api2vec = self.matrix_load(flags.api2vec)
        # self.train_dataset = self.get_train_data.dataset
        # self.valid_dataset = self.get_valid_data.dataset
        self.test_dataset = self.get_data.dataset
        self.word2id = self.get_data.word2id

        self.graph_iteration = 2
        self.node_iteration = 2
        self.graph_dim = 64




        self.train_loss = []
        self.valid_loss = []
        self.min_valid_loss = np.inf
        self.val_auc = np.inf
        self.BEST_VAL_AUC = 0
        self.BEST_TEST_AUC = 0

    def matrix_load(self, id2vec):
        matrix = np.load(id2vec)
        return matrix


    def build_search_dataset(self):

        # model_name = "/home/yuqing/xiabing/BinaryEmbedding/X2V_graph_with_kb.model"
        model_name = self.best_network_model
        # best_model=torch.load(model_name).get('epoch')
        # print(best_model)
        # exit(0)
        best_model = torch.load(model_name).get('model').cuda()
        best_model.eval()

        final_predict = []
        ground_truth = []
        search_funcs_vec = list()

        test_dataset = self.test_dataset
        dataset_len = len(test_dataset)
        iterations = int(dataset_len / self.batch_size)

        for itera in range(0, iterations):
            if self.network.startswith('x2v'):
                inputs, lable ,funcs = self.get_x2v_inputs(itera, test_dataset)

            predictions, F1_embedding, F2_embedding = best_model(inputs)

            pairs = zip(funcs,F1_embedding)

            g_truth = lable.cpu().detach().numpy()

            final_pred = predictions.cpu().detach().numpy()

            ground_truth.extend(g_truth)
            final_predict.extend(final_pred)

            for funa,funvec in pairs:
                a_function_vec = dict()
                a_function_vec['file_name']= funa['file_name']
                a_function_vec['fun_name'] = funa['fun_name']
                a_function_vec['fun_address'] = funa['fun_address']
                # a_function_vec['fun_cfg'] = funa['cfg']
                a_function_vec['fun_vec'] = funvec.cpu().detach().numpy().tolist()
                search_funcs_vec.append(a_function_vec)
                print(a_function_vec['fun_name']+'-->done')

        fun_database = './X2V_ALKG_kb_search_funcs_vec.json'
        random.shuffle(search_funcs_vec)

        with open(fun_database, 'w') as basefile:
            json.dump(search_funcs_vec, basefile, indent=4, ensure_ascii=False)
        print(fun_database + ':' + 'done')

    def get_x2v_inputs(self,itera,dataset):

        start = itera * self.batch_size
        end = (itera + 1) * self.batch_size
        # print(start, end)
        funcs = dataset[start:end]

        # inputs = [g1_inputs,g2_inputs,lable,g1_api,g2_api,g1_length,
        # g2_length,g1_api_length,g2_api_length,g1_acfg,g2_acfg]
        seq_pair = self.get_data.get_x2v_seq_singles(funcs)
        seq_g1_inputs = torch.tensor(seq_pair[0])
        seq_g2_inputs = torch.tensor(seq_pair[1])

        # graph_pair = [g1_adj, g1_matrix, g1_length,g2_adj, g2_matrix, g2_length,lable,
        # g1_api, g2_api, g1_api_length, g2_api_length，ga_api,gb_api]
        graph_pair = self.get_data.get_graph_singles(funcs)

        g1_adj = torch.tensor(graph_pair[0])
        # g1_inputs =[batch,max_nodes,max_instrutions]--->[batch,seq]
        g1_inputs = torch.tensor(graph_pair[1])
        g1_inputs = g1_inputs.reshape(-1, self.max_instructions)

        g2_adj = torch.tensor(graph_pair[3])

        g2_inputs = torch.tensor(graph_pair[4])
        g2_inputs = g2_inputs.reshape(-1, self.max_instructions)

        lable = torch.Tensor(graph_pair[6]).to(DEVICE)

        ga_api = torch.tensor(graph_pair[11])
        gb_api = torch.tensor(graph_pair[12])

        # inputs = [g1_adj, g1_matrix, g1_length,g2_adj, g2_matrix, g2_length,lable]
        inputs = [seq_g1_inputs, seq_g2_inputs, g1_adj, g1_inputs, g2_adj, g2_inputs, lable, ga_api, gb_api]

        return inputs, lable, funcs




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--word2id", type=str, help="word2id to analyze")
    parser.add_argument("-v", "--id2vec", type=str, help="id2vec to analyze")
    # parser.add_argument("-x", "-train_dataset", type=str, help="train_dataset for analyze")
    # parser.add_argument("-y", "--valid_dataset", type=str, help="train_dataset for analyze")
    parser.add_argument("-z", "--search_dataset", type=str, help="train_dataset for analyze")
    parser.add_argument("-n", "--network", type=str, help="network for analyze")
    parser.add_argument("-m", "--model", type=str, help="model network for test")

    parser.add_argument("-ai", "--api2id", type=str, help="word2id to analyze")
    parser.add_argument("-av", "--api2vec", type=str, help="id2vec to analyze")
    args = parser.parse_args()
    # args.network: Graph_Transformer\Graph_BiLSTM\seq_Attention_LSTM\seq_LSTM
    print(time.asctime(time.localtime(time.time())))

    trainer = x2vTrainer(args)

    trainer.build_search_dataset()

    print(time.asctime(time.localtime(time.time())))
    # trainer.scan_data()
    # trainer.test()
    # trainer.static_data()