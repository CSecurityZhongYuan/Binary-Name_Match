import os
import json
os.environ['CUDA_VISIBLE_DEVICES']='0,1'
import torch.backends.cudnn as cudnn
import argparse
import torch
from torch import nn

import random
import time
from datetime import datetime

from sklearn import metrics
import numpy as np
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from getPairsDataset import pairsDataset

# from BinaryEmbedding.Model.seqLSTMAttentionNetwork import seqLSTMAttentionNetwork
#
# from BinaryEmbedding.Model.seqGRU import seqGRUNetwork
#
# from BinaryEmbedding.Model.graphLSTMNetwork import graphLSTMNetwork
#
# from BinaryEmbedding.Model.GraphTransformerWithKB import  graphLSTMCatNetwork
# from BinaryEmbedding.Model.acfgGraphNetwork import  acfgGraphNetwork
# from BinaryEmbedding.Model.acfgSeqNetwork import acfgSeqLSTMAttentionNetwork



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
        # self.network = flags.network
        self.flag = 3
        self.basename= os.path.basename(flags.test_dataset)


        if self.flag == 1:
            # X2V
            self.batch_size = 100
            flags.knowledge='True'
        if self.flag == 2:
            # SAFE
            self.batch_size = 250
            flags.knowledge = 'False'

        if self.flag == 3:
            # UFEBA
            self.batch_size = 50
            flags.knowledge = 'False'


        self.epochs = 10
        self.max_instructions = 50
        self.max_nodes = 150

        # seq
        self.seq_len = 150
        # kb seq length for all basic block with cat( func= graph + kb)
        self.kb_seq = 150
        # graph for node kb length
        self.node_kb_max = 5


        # self.lstm_batch_size = self.batch_size * self.max_nodes
        self.lr = 0.001

        # i2v
        self.embedding_dim = 100
        self.hidden_dim = 100

        self.output_dim = 64
        self.num_layers = 1

        self.n_heads = 4

        if flags.knowledge=='True':
            self.with_kb = True
            self.cat_kb = 'kb'
            # print('with kb,func=func+kb')
        else:
            self.with_kb = False
            self.cat_kb = 'no_kb'
            # print('with no kb')

        # acfg
        # self.acfg_length = 8
        self.acfg_length = 7


        self.get_data = pairsDataset(flags.word2id, flags.api2id,flags.test_dataset,self.seq_len,
                                          self.max_instructions,
                                          self.max_nodes,
                                          self.kb_seq,
                                          self.node_kb_max,
                                          self.max_path,
                                          self.path_len
                                          )

        self.id2vec = self.matrix_load(flags.id2vec)
        self.api2vec = self.matrix_load(flags.api2vec)

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

    def get_seq_inputs(self,itera,dataset):
        start = itera*self.batch_size
        end = (itera+1)*self.batch_size
        # print(start, end)
        pairs = dataset[start:end]
        # g1_matrix, g2_matrix, lable,g1_api,g2_api,g1_length,g2_length,g1_api_length,g2_api_length

        pair = self.get_data.get_pairs(pairs)

        g1_inputs = torch.tensor(pair[0])

        g2_inputs = torch.tensor(pair[1])
        lable = torch.Tensor(pair[2]).to(DEVICE)
        g1_api = torch.tensor(pair[3])
        g2_api = torch.tensor(pair[4])
        g1_length = torch.tensor(pair[5])
        g2_length = torch.tensor(pair[6])
        g1_api_length = torch.tensor(pair[7])
        g2_api_length = torch.tensor(pair[8])



        inputs = [g1_inputs,g2_inputs,lable,g1_api,g2_api,g1_length,g2_length,g1_api_length,g2_api_length]

        return inputs,lable

    def get_inputs(self,itera,dataset):
        start = itera*self.batch_size
        end = (itera+1)*self.batch_size
        # print(start, end)
        pairs = dataset[start:end]

        # pair = [g1_adj, g1_matrix, g1_length,g2_adj, g2_matrix, g2_length,lable,
        # g1_api, g2_api, g1_api_length, g2_api_length，ga_api,gb_api]
        pair = self.get_data.get_batch_pairs(pairs)

        g1_adj = torch.tensor(pair[0])
        # g1_inputs =[batch,max_nodes,max_instrutions]--->[batch,seq]
        g1_inputs = torch.tensor(pair[1])
        g1_inputs= g1_inputs.reshape(-1, self.max_instructions)

        g1_length = torch.tensor(pair[2])
        g2_adj = torch.tensor(pair[3])

        g2_inputs = torch.tensor(pair[4])
        g2_inputs = g2_inputs.reshape(-1, self.max_instructions)

        g2_length = torch.tensor(pair[5])
        lable = torch.Tensor(pair[6]).to(DEVICE)

        # g1_inputs =[batch,max_nodes,max_instrutions]--->[batch,seq]
        g1_api = torch.tensor(pair[7])

        g1_api = g1_api.reshape(-1, self.node_kb_max)
        g2_api = torch.tensor(pair[8])
        g2_api = g2_api.reshape(-1, self.node_kb_max)


        g1_api_length = torch.tensor(pair[9])
        g2_api_length = torch.tensor(pair[10])

        ga_api = torch.tensor(pair[11])
        gb_api = torch.tensor(pair[12])

        # inputs = [g1_adj, g1_matrix, g1_length,g2_adj, g2_matrix, g2_length,lable]
        inputs = [g1_adj, g1_inputs, g1_length, g2_adj, g2_inputs, g2_length,lable,
                  g1_api, g2_api, g1_api_length, g2_api_length,ga_api,gb_api]

        return inputs,lable
    def model_test(self):

        # best_model=torch.load(model_name).get('epoch')
        # print(best_model)

        if self.flag == 1:
            # X2V
            model_name = '/home/yuqing/xiabing/BinaryEmbedding/best_val_auc_model/best_val_auc_Graph_LSTM_With_KB_kb.model'
            network = 'Graph_X2V'

        if self.flag == 2:
            # SAFE
            model_name = '/home/yuqing/xiabing/BinaryEmbedding/best_val_auc_model/best_val_auc_seq_GRU_no_kb.model'
            network = 'seq_SAFE'

        if self.flag == 3:
            # I2V
            model_name = '/home/yuqing/xiabing/BinaryEmbedding/best_val_auc_model/best_val_auc_Graph_BiLSTM_no_kb.model'
            network = 'Graph_I2V'





        best_model = torch.load(model_name).get('model').cuda()
        best_model.eval()

        final_predict = []
        ground_truth = []

        test_dataset = self.test_dataset
        dataset_len = len(test_dataset)
        iterations = int(dataset_len / self.batch_size)

        for itera in range(0, iterations):
            if network.startswith('seq'):
                inputs, lable = self.get_seq_inputs(itera, test_dataset)
            if network.startswith('Graph'):
                inputs,lable = self.get_inputs(itera,test_dataset)

            predictions, F1_embedding, F2_embedding = best_model(inputs)

            g_truth = lable.cpu().detach().numpy()

            final_pred = predictions.cpu().detach().numpy()

            ground_truth.extend(g_truth)
            final_predict.extend(final_pred)





        test_fpr, test_tpr, test_thresholds = metrics.roc_curve(ground_truth, final_predict, pos_label=1)
        test_auc = metrics.auc(test_fpr, test_tpr)
        # if test_auc>self.BEST_TEST_AUC:
        #     self.BEST_TEST_AUC = test_auc
        #     fig = plt.figure()
        #     plt.title('Receiver Operating Characteristic')
        #     plt.plot(test_fpr, test_tpr, 'b', label='AUC = %0.2f' % test_auc)
        #     fig.savefig("./best_test_roc.png")
        #     print("\nNEW BEST_VAL_AUC: {} !\n\ttest_auc : {}\n".format(self.BEST_VAL_AUC, test_auc))
        #     plt.close(fig)

        log_string = (self.basename+'_'+network+'**Model openssl***,test_auc:{:0.6f}  '.format(test_auc))
        print(log_string)
        self.log('./model_test.log', log_string)


    def log(self,filename,s):
        f = open(filename,'a')
        f.write(str(str(time.asctime(time.localtime(time.time())))+':'+s+'\n'))
        f.close()
def scan_for_file(start):
    file_list = []
    for root, dirs, files in os.walk(start):
        for file in files:
            file_list.append(os.path.join(start, file))
    return file_list
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--word2id", type=str, help="word2id to analyze")
    parser.add_argument("-v", "--id2vec", type=str, help="id2vec to analyze")
    parser.add_argument("-z", "--test_dataset", type=str, help="train_dataset for analyze")
    parser.add_argument("-ai", "--api2id", type=str, help="word2id to analyze")
    parser.add_argument("-av", "--api2vec", type=str, help="id2vec to analyze")

    args = parser.parse_args()


    print(time.asctime(time.localtime(time.time())))
    file_list = scan_for_file(args.test_dataset)
    for files in file_list:
        args.test_dataset = files
        trainer = x2vTrainer(args)
        trainer.model_test()

    print(time.asctime(time.localtime(time.time())))
