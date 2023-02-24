import os
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

from DataSet.pairsDataset import pairsDataset


# SAFE model
from Model.seqGRU import seqGRUNetwork
# EFEBA model
from Model.graphLSTMNetwork import graphLSTMNetwork
# Name_Match model
from Model.GraphLSTMWithKB import  graphLSTMCatNetwork
# Gemini model
from Model.acfgGraphNetwork import  acfgGraphNetwork




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
        self.epochs = 10
        self.batch_size = 250

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

        if flags.knowledge=='True':
            self.with_kb = True
            self.cat_kb = 'kb'
            print('with kb,func=func+kb')
        else:
            self.with_kb = False
            self.cat_kb = 'no_kb'
            print('with no kb')

        # acfg
        self.acfg_length = 7


        self.get_train_data = pairsDataset(flags.word2id, flags.api2id,flags.train_dataset,
                                           self.seq_len,
                                           self.max_instructions,
                                           self.max_nodes,
                                           self.kb_seq,
                                           self.node_kb_max,
                                           self.max_path,
                                           self.path_len)
        self.get_valid_data = pairsDataset(flags.word2id,flags.api2id, flags.valid_dataset,self.seq_len,
                                           self.max_instructions,
                                           self.max_nodes,
                                           self.kb_seq,
                                           self.node_kb_max,
                                           self.max_path,
                                           self.path_len
                                           )
        self.get_test_data = pairsDataset(flags.word2id, flags.api2id,flags.test_dataset,self.seq_len,
                                          self.max_instructions,
                                          self.max_nodes,
                                          self.kb_seq,
                                          self.node_kb_max,
                                          self.max_path,
                                          self.path_len
                                          )

        self.id2vec = self.matrix_load(flags.id2vec)
        self.api2vec = self.matrix_load(flags.api2vec)
        self.train_dataset = self.get_train_data.dataset
        self.valid_dataset = self.get_valid_data.dataset
        self.test_dataset = self.get_test_data.dataset
        self.word2id = self.get_train_data.word2id

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

        pair = self.get_train_data.get_seq_pairs(pairs)

        g1_inputs = torch.tensor(pair[0])

        g2_inputs = torch.tensor(pair[1])
        lable = torch.Tensor(pair[2]).to(DEVICE)
        g1_api = torch.tensor(pair[3])
        g2_api = torch.tensor(pair[4])
        g1_length = torch.tensor(pair[5])
        g2_length = torch.tensor(pair[6])
        g1_api_length = torch.tensor(pair[7])
        g2_api_length = torch.tensor(pair[8])



        # inputs = [g1_adj, g1_matrix, g1_length,g2_adj, g2_matrix, g2_length,lable]
        inputs = [g1_inputs,g2_inputs,lable,g1_api,g2_api,g1_length,g2_length,g1_api_length,g2_api_length]

        return inputs,lable

    def get_inputs(self,itera,dataset):
        start = itera*self.batch_size
        end = (itera+1)*self.batch_size
        # print(start, end)
        pairs = dataset[start:end]

        # pair = [g1_adj, g1_matrix, g1_length,g2_adj, g2_matrix, g2_length,lable,
        # g1_api, g2_api, g1_api_length, g2_api_length，ga_api,gb_api]
        pair = self.get_train_data.get_graph_pairs(pairs)

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

    def get_acfg_inputs(self,itera,dataset):
        start = itera*self.batch_size
        end = (itera+1)*self.batch_size
        # print(start, end)
        pairs = dataset[start:end]

        # pair = [g1_adj, g1_matrix, g1_length,g2_adj, g2_matrix, g2_length,lable,
        # g1_api, g2_api, g1_api_length, g2_api_length,ga_api,gb_api]
        pair = self.get_train_data.get_acfg_pairs(pairs)

        g1_adj = torch.tensor(pair[0])
        # g1_inputs =[batch,max_nodes,max_instrutions]--->[batch,seq]
        g1_inputs = torch.tensor(pair[1])

        g1_inputs= g1_inputs.reshape(-1, self.acfg_length)


        g1_length = torch.tensor(pair[2])
        g2_adj = torch.tensor(pair[3])

        g2_inputs = torch.tensor(pair[4])
        g2_inputs = g2_inputs.reshape(-1, self.acfg_length)

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
        inputs = [g1_adj, g1_inputs, g2_adj, g2_inputs, g1_api, g2_api,ga_api,gb_api]

        return inputs,lable


    def train(self,epoch):
        train_dataset = self.train_dataset

        random.shuffle(train_dataset)
        dataset_len = len(train_dataset)
        iterations = int(dataset_len / self.batch_size)
        total_train_loss = []
        self.NET.train()

        for itera in range(0, iterations):
            self.optimizer.zero_grad()
            if self.network.startswith('seq'):
                inputs, lable = self.get_seq_inputs(itera, train_dataset)
            if self.network.startswith('Graph'):
                inputs,lable = self.get_inputs(itera,train_dataset)
            if self.network.startswith('acfg'):
                inputs,lable = self.get_acfg_inputs(itera,train_dataset)



            predictions,F1_embedding, F2_embedding= self.NET(inputs)
            loss = self.loss_func(predictions, lable).to(DEVICE)

            loss.backward()

            self.optimizer.step()

            if itera%100==0:
                print('epoch:',epoch+1,'itera:',itera,'loss:',loss.item())


            total_train_loss.append(loss.item())

        self.train_loss.append(np.mean(total_train_loss))

    def test(self):
        # best_model = torch.load('./LSTM.model').get('model').cuda()
        # best_model = torch.load('./Transformer.model').get('model').cuda()
        # best_model.eval()
        self.NET.eval()
        final_predict = []
        ground_truth = []

        test_dataset = self.test_dataset
        dataset_len = len(test_dataset)
        iterations = int(dataset_len / self.batch_size)
        total_valid_loss = []

        for itera in range(0, iterations):
            if self.network.startswith('seq'):
                inputs, lable = self.get_seq_inputs(itera, test_dataset)
            if self.network.startswith('Graph'):
                inputs,lable = self.get_inputs(itera,test_dataset)
            if self.network.startswith('acfg'):
                inputs,lable = self.get_acfg_inputs(itera,test_dataset)

            predictions, F1_embedding, F2_embedding = self.NET(inputs)

            g_truth = lable.cpu().detach().numpy()

            final_pred = predictions.cpu().detach().numpy()

            ground_truth.extend(g_truth)
            final_predict.extend(final_pred)

            loss = self.loss_func(predictions, lable).to(DEVICE)
            if itera%10==0:
                print('itera:',itera,'loss:',loss.item())

        test_fpr, test_tpr, test_thresholds = metrics.roc_curve(ground_truth, final_predict, pos_label=1)
        test_auc = metrics.auc(test_fpr, test_tpr)

        # predict = final_predict.astype(np.int64)
        #
        # test_precision = metrics.precision_score(ground_truth, predict, pos_label=1)
        # test_recall = metrics.recall_score(ground_truth, predict, pos_label=1)
        # test_f1 = metrics.f1_score(ground_truth, predict, pos_label=1)
        if test_auc>self.BEST_TEST_AUC:
            self.BEST_TEST_AUC = test_auc
            fig = plt.figure()
            plt.title('Receiver Operating Characteristic')
            plt.plot(test_fpr, test_tpr, 'b', label='AUC = %0.2f' % test_auc)
            fig.savefig('./comFU_W_new'+'_'+self.network+'_'+self.cat_kb+'.png')
            print("\nNEW BEST_VAL_AUC: {} !\n\ttest_auc : {}\n".format(self.BEST_VAL_AUC, test_auc))
            np.savez('./comFU_W_new' + '_' + self.network + '_' + self.cat_kb + '.result', fpr=test_fpr,
                     tpr=test_tpr, auc=self.BEST_TEST_AUC, ground =ground_truth,predict=final_predict )
            plt.close(fig)
            # np.savez('./comFU_W_new'+'_'+self.network+'_'+self.cat_kb+'.result',fpr=test_fpr,
            #          tpr=test_tpr,auc=self.BEST_TEST_AUC,precision= test_precision,recall=test_recall,f_scroe=test_f1)
            # plt.close(fig)

        log_string = (
        'TEST***,train_loss:{:0.6f},valid_loss:{:0.6f},best_valid_loss:{:0.6f},best_valid_auc:{:0.6f},best_test_auc:{:0.6f}'.format(
            self.train_loss[-1],
            self.valid_loss[-1],
            self.min_valid_loss,
            self.BEST_VAL_AUC,
            self.BEST_TEST_AUC,)
        )

        print(log_string)
        self.log('./LSTM.log', log_string)

    def valid(self,epoch):
        valid_dataset = self.valid_dataset
        random.shuffle(valid_dataset)
        dataset_len = len(valid_dataset)
        iterations = int(dataset_len / self.batch_size)
        total_valid_loss = []

        final_predict = []
        ground_truth = []
        self.NET.eval()

        for itera in range(0, iterations):
            if self.network.startswith('seq'):
                inputs, lable = self.get_seq_inputs(itera, valid_dataset)
            if self.network.startswith('Graph'):
                inputs,lable = self.get_inputs(itera,valid_dataset)
            if self.network.startswith('acfg'):
                inputs,lable = self.get_acfg_inputs(itera,valid_dataset)

            predictions, F1_embedding, F2_embedding = self.NET(inputs)
            loss = self.loss_func(predictions,lable).to(DEVICE)

            if itera % 10 == 0:
                print('epoch:', epoch+1, 'itera:', itera, 'loss:', loss.item())

            g_truth = lable.cpu().detach().numpy()
            final_pred = predictions.cpu().detach().numpy()
            ground_truth.extend(g_truth)
            final_predict.extend(final_pred)
            total_valid_loss.append(loss.item())

        self.valid_loss.append(np.mean(total_valid_loss))

        val_fpr, val_tpr, val_thresholds = metrics.roc_curve(ground_truth, final_predict, pos_label=1)
        val_auc = metrics.auc(val_fpr, val_tpr)
        print(time.asctime(time.localtime(time.time())))
        print('epoch:', (epoch+1),  'val_auc:', val_auc)
        self.val_auc = val_auc

        if self.val_auc > self.BEST_VAL_AUC:
            torch.save({'epoch':epoch,'model':self.NET,'train_loss':self.train_loss,
                        'valid_loss':self.valid_loss,'val_auc':self.val_auc},'./comFU_W_new'+
                       '_'+self.network+'_'+self.cat_kb+'.model')
            self.BEST_VAL_AUC = self.val_auc

            self.test()

        if self.valid_loss[-1] < self.min_valid_loss:
            self.min_valid_loss = self.valid_loss[-1]

        log_string = ('iter:[{:d}/{:d}],train_loss:{:0.6f},valid_loss:{:0.6f},'
                      'best_valid_loss:{:0.6f},best_valid_auc:{:0.6f},'
                      'best_test_auc:{:0.6f},lr:{:0.7f}'.format(
            (epoch+1),
            self.epochs,
            self.train_loss[-1],
            self.valid_loss[-1],
            self.min_valid_loss,
            self.BEST_VAL_AUC,
            self.BEST_TEST_AUC,
            self.optimizer.param_groups[0]['lr']))

        # self.multi_step_scheduler.step()

        print(log_string)
        self.log('./LSTM.log',log_string)
    def log(self,filename,s):
        f = open(filename,'a')
        f.write(str(str(time.asctime(time.localtime(time.time())))+':'+s+'\n'))
        f.close()

    def get_network_type(self):
        # A: Attention L:LSTM G:GNN K:Knowledge
        # Name_Match
        if self.network == 'Graph_Name_Match':
            self.NET = graphLSTMCatNetwork(self.id2vec,
                                            self.api2vec,
                                            self.embedding_dim,
                                            self.hidden_dim,
                                            self.num_layers,
                                            self.max_nodes,
                                            self.batch_size,
                                            self.output_dim,
                                            self.with_kb
                                          ).to(DEVICE)
            self.optimizer = torch.optim.Adam(self.NET.parameters(), lr=self.lr)
            self.loss_func = nn.MSELoss()

            # self.multi_step_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=2, gamma=0.1)
            if USE_CUDA:
                self.NET = self.NET.cuda()
        # Gemini
        if self.network =='acfg_Gemini':
            self.NET = acfgGraphNetwork(self.id2vec,
                                        self.api2vec,
                                   self.embedding_dim,
                                   self.hidden_dim,
                                   self.num_layers,
                                   self.max_nodes,
                                   self.batch_size,
                                   self.output_dim,
                                   self.with_kb).to(DEVICE)
            self.optimizer = torch.optim.Adam(self.NET.parameters(), lr=self.lr)
            # self.loss_func = ContrastiveLoss()
            self.loss_func = nn.MSELoss()
            if USE_CUDA:
                self.NET = self.NET.cuda()
        #EFEBA
        if self.network =='Graph_EFEBA':
            self.NET = graphLSTMNetwork(self.id2vec,
                                        self.api2vec,
                                   self.embedding_dim,
                                   self.hidden_dim,
                                   self.num_layers,
                                   self.max_nodes,
                                   self.batch_size,
                                   self.output_dim,
                                   self.with_kb).to(DEVICE)
            self.optimizer = torch.optim.Adam(self.NET.parameters(), lr=self.lr)
            # self.loss_func = ContrastiveLoss()
            self.loss_func = nn.MSELoss()
            # self.loss_func = ContrastiveLoss(margin=5.0)
            # self.multi_step_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
            #                                                                  milestones=[2, 4, 6, 8, 10],
            #                                                                  gamma=0.1)
            # self.multi_step_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=2,gamma=0.1)
            if USE_CUDA:
                self.NET = self.NET.cuda()
        #SAFE
        # SAFE
        if self.network == 'seq_SAFE':
            self.NET = seqGRUNetwork(self.id2vec,
                                        self.api2vec,
                                        self.output_dim,
                                        self.embedding_dim,
                                        self.hidden_dim,
                                        self.num_layers,
                                        self.max_nodes,
                                        self.batch_size,
                                        self.with_kb).to(DEVICE)
            self.optimizer = torch.optim.Adam(self.NET.parameters(), lr=self.lr)
            self.loss_func = nn.MSELoss()
            # self.multi_step_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=2, gamma=0.1)

            if USE_CUDA:
                self.NET = self.NET.cuda()

    def do(self):
        self.get_network_type()
        lrs,epochs =[],[]
        for epoch in range(0, self.epochs):
            # lrs.append(self.multi_step_scheduler.get_lr())
            lrs.append((self.optimizer.param_groups[0]['lr']))
            epochs.append(epoch)
            # -------------
            # Train
            # ------------
            print('---Training  %s epoch start----\n'%(epoch+1),time.asctime(time.localtime(time.time())))
            self.train(epoch)

            print('---Training %s epoch end,start valid ----\n'%(epoch+1),time.asctime(time.localtime(time.time())))
            # -------------
            # Valid
            # ------------
            self.valid(epoch)
            print('---Valid  %s epoch end----\n' % (epoch+1), time.asctime(time.localtime(time.time())))


        np.savez('./comFU_W_new'+'_loss' + '_' + self.network + '_' + self.cat_kb + '.result',
                 train_loss=self.train_loss, valid_loss=self.valid_loss)


    def model_test(self):
        model_name = './best_val_auc_seq_Attention_LSTM_kb.model'
        best_model = torch.load(model_name).get('model').cuda()
        best_model.eval()

        final_predict = []
        ground_truth = []

        test_dataset = self.test_dataset
        dataset_len = len(test_dataset)
        iterations = int(dataset_len / self.batch_size)
        total_valid_loss = []

        for itera in range(0, iterations):
            if self.network.startswith('seq'):
                inputs, lable = self.get_seq_inputs(itera, test_dataset)
            if self.network.startswith('Graph'):
                inputs,lable = self.get_inputs(itera,test_dataset)

            # with torch.no_grad():
                # predictions = best_model(inputs).to(DEVICE)
                # predictions = best_model(inputs)
            # with torch.no_grad():
            #     predictions ,F1_embedding, F2_embedding= self.NET(inputs)

            predictions, F1_embedding, F2_embedding = best_model(inputs)

            g_truth = lable.cpu().detach().numpy()

            final_pred = predictions.cpu().detach().numpy()

            ground_truth.extend(g_truth)
            final_predict.extend(final_pred)

            loss = self.loss_func(predictions, lable).to(DEVICE)
            if itera%10==0:
                print('itera:',itera,'loss:',loss.item())


        test_fpr, test_tpr, test_thresholds = metrics.roc_curve(ground_truth, final_predict, pos_label=1)
        test_auc = metrics.auc(test_fpr, test_tpr)
        if test_auc>self.BEST_TEST_AUC:
            self.BEST_TEST_AUC = test_auc
            fig = plt.figure()
            plt.title('Receiver Operating Characteristic')
            plt.plot(test_fpr, test_tpr, 'b', label='AUC = %0.2f' % test_auc)
            fig.savefig("./best_test_roc.png")
            print("\nNEW BEST_VAL_AUC: {} !\n\ttest_auc : {}\n".format(self.BEST_VAL_AUC, test_auc))
            plt.close(fig)

        log_string = (
        'TEST***,train_loss:{:0.6f},valid_loss:{:0.6f},best_valid_loss:{:0.6f},best_valid_auc:{:0.6f},best_test_auc:{:0.6f}'.format(
            self.train_loss[-1],
            self.valid_loss[-1],
            self.min_valid_loss,
            self.BEST_VAL_AUC,
            self.BEST_TEST_AUC,)
        )

        print(log_string)
        self.log('./LSTM.log', log_string)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--word2id", type=str, help="word2id to analyze")
    parser.add_argument("-v", "--id2vec", type=str, help="id2vec to analyze")
    parser.add_argument("-x", "--train_dataset", type=str, help="train_dataset for analyze")
    parser.add_argument("-y", "--valid_dataset", type=str, help="train_dataset for analyze")
    parser.add_argument("-z", "--test_dataset", type=str, help="train_dataset for analyze")
    parser.add_argument("-n", "--network", type=str, help="network for analyze")
    parser.add_argument("-k", "--knowledge", type=str, help="cat knowledge for analyze")
    parser.add_argument("-ai", "--api2id", type=str, help="word2id to analyze")
    parser.add_argument("-av", "--api2vec", type=str, help="id2vec to analyze")
    args = parser.parse_args()
    # args.network: Graph_Transformer\Graph_BiLSTM\seq_Attention_LSTM\seq_LSTM
    print(time.asctime(time.localtime(time.time())))

    trainer = x2vTrainer(args)

    trainer.do()

    print(time.asctime(time.localtime(time.time())))
