import time
import argparse
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import numpy as np

torch.manual_seed(1234)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class graphLSTMCatNetwork(nn.Module):

    # @staticmethod
    # def weight_init(m):
    #     if isinstance(m,nn.Linear):
    #         nn.init.xavier_normal_(m.weight)
    #         nn.init.constant_(m.bias,0)
    #     elif isinstance(m,nn.BatchNorm1d):
    #         nn.init.constant_(m.weight,1)
    #         nn.init.constant_(m.bias,0)

    def __init__(self,id2vec,api2vec,embedding_dim,hidden_dim,num_layers,max_nodes,batch_size,out_dim,with_kb):
        super(graphLSTMCatNetwork,self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_size = batch_size * max_nodes

        self.graph_num = batch_size
        self.max_nodes = max_nodes

        self.graph_iteration = 2
        self.node_iteration = 2
        self.graph_dim = out_dim

        self.dense_dim = 100
        self.with_kb = with_kb

        # self.id2vec = self.matrix_load(id2vec)
        self.id2vec = id2vec
        self.kb2vec = api2vec
        self.i_encoder = nn.LSTM(input_size= self.embedding_dim,
                               hidden_size= self.hidden_dim,
                               num_layers= self.num_layers,
                               batch_first=False).to(DEVICE)

        self.kb_encoder = nn.LSTM(input_size=self.embedding_dim,
                                 hidden_size=self.hidden_dim,
                                 num_layers=self.num_layers,
                                 batch_first=False).to(DEVICE)
        # for name,param in self.i_encoder.named_parameters():
        #     nn.init.uniform_(param,0,1)
        # for name, param in self.kb_encoder.named_parameters():
        #     nn.init.uniform_(param, 0, 1)

        self.i_fc1 = nn.Sequential(nn.Linear(self.hidden_dim, self.dense_dim,bias=True))

        self.i_fc3 = nn.Sequential(nn.Linear(self.dense_dim, self.dense_dim, bias=True))

        self.kb_fc1 = nn.Sequential(nn.Linear(self.hidden_dim, self.dense_dim, bias=True))

        self.kb_fc3 = nn.Sequential(nn.Linear(self.dense_dim, self.dense_dim, bias=True))

        # self.fc1 = nn.Sequential(nn.Linear(self.hidden_dim * 2, self.dense_dim, bias=True))
        #
        # self.fc2 = nn.Sequential(nn.Linear(self.dense_dim, self.output_dim, bias=True))
        #
        self.fc2 = nn.Sequential(nn.Linear(self.graph_dim, self.graph_dim))

        self.kbcat_fc1 = nn.Sequential(nn.Linear(self.hidden_dim , self.dense_dim, bias=True))

        self.kbcat_fc2 = nn.Sequential(nn.Linear(self.dense_dim, self.graph_dim, bias=True))


        # one-hot
        # self.i_embedding =  nn.Embedding(self.id2vec.shape[0], embedding_dim)
        # self.kb_embedding = nn.Embedding(self.kb2vec.shape[0], embedding_dim)

        # x2v
        self.i_embedding = nn.Embedding(self.id2vec.shape[0], embedding_dim)
        self.i_embedding.weight = nn.Parameter(torch.tensor(self.id2vec, dtype=torch.float32))
        self.i_embedding.weight.requires_grad = False

        # k2v
        self.kb_embedding = nn.Embedding(self.kb2vec.shape[0], embedding_dim)
        self.kb_embedding.weight = nn.Parameter(torch.tensor(self.kb2vec, dtype=torch.float32))
        self.kb_embedding.weight.requires_grad = False

        # self.a_hidden = self.init_hidden()
        # self.b_hidden = self.init_hidden()
        self.w1,self.w2,self.w_n = self.init_graph_weight()

        self.i_w_omega = nn.Parameter(torch.Tensor(self.dense_dim, self.dense_dim))
        self.i_u_omega = nn.Parameter(torch.Tensor(self.dense_dim, 1))
        nn.init.uniform_(self.i_w_omega, -0.1, 0.1)
        nn.init.uniform_(self.i_u_omega, -0.1, 0.1)
        # nn.init.uniform_(self.i_w_omega, 0, 1)
        # nn.init.uniform_(self.i_u_omega, 0, 1)

        self.kb_w_omega = nn.Parameter(torch.Tensor(self.dense_dim, self.dense_dim))
        self.kb_u_omega = nn.Parameter(torch.Tensor(self.dense_dim, 1))
        nn.init.uniform_(self.kb_w_omega, -0.1, 0.1)
        nn.init.uniform_(self.kb_u_omega, -0.1, 0.1)
        # nn.init.uniform_(self.kb_w_omega, 0, 1)
        # nn.init.uniform_(self.kb_u_omega, 0, 1)

    def attention_lstm(self,x):

        u = torch.tanh(torch.matmul(x,self.i_w_omega))
        att = torch.matmul(u,self.i_u_omega)

        att_score = F.softmax(att,dim=1)
        # print(att_score.size())
        scored_x = x * att_score
        # print(scored_x.size())
        att_out = torch.sum(scored_x,dim=1)
        # print(att_out.size())
        return att_out
    def attention_kb(self,x):

        u = torch.tanh(torch.matmul(x,self.kb_w_omega))
        att = torch.matmul(u,self.kb_u_omega)

        att_score = F.softmax(att,dim=1)
        # print(att_score.size())
        scored_x = x * att_score
        # print(scored_x.size())
        att_out = torch.sum(scored_x,dim=1)
        # print(att_out.size())
        return att_out
    def init_graph_weight(self):
        w1 = torch.randn(self.dense_dim, self.graph_dim).to(DEVICE)
        w2 = torch.randn(self.max_nodes, self.graph_dim).to(DEVICE)
        w_n = []
        for i in range(0, self.graph_iteration):
            w_n.append(torch.randn(self.graph_dim, self.graph_dim).to(DEVICE))
        return w1,w2,w_n

    # def init_hidden(self):
    #     return (torch.zeros(1,self.batch_size,self.hidden_dim).to(DEVICE),
    #             torch.zeros(1,self.batch_size,self.hidden_dim).to(DEVICE))

    def forward(self,inputs):

        # inputs = [g1_adj, g1_inputs, g1_length, g2_adj, g2_inputs, g2_length,lable,
        #                   g1_api, g2_api, g1_api_length, g2_api_length,ga_api,gb_api]
        #         g1_adj = torch.tensor(pair[0])
        #         g1_inputs = torch.tensor(pair[1])
        #         g1_length = torch.tensor(pair[2])
        #         g2_adj = torch.tensor(pair[3])
        #         g2_inputs = torch.tensor(pair[4])
        #         g2_length = torch.tensor(pair[5])
        #         lable = torch.Tensor(pair[6]).to(DEVICE)
        #         g1_api = torch.tensor(pair[7])
        #         g2_api = torch.tensor(pair[8])
        #         g1_api_length = torch.tensor(pair[9])
        #         g2_api_length = torch.tensor(pair[10])
        if self.with_kb:
            a_last_out, b_last_out = self.get_i_embed(inputs[0],inputs[1],inputs[3],inputs[4])
            # add kb with graph
            # a_api_out, b_api_out = self.get_kbadd_embed(inputs[0],inputs[7],inputs[3],inputs[8])

            # cat kb with seq
            a_api_out, b_api_out = self.get_kbcat_embed(inputs[11], inputs[12])


            # A = torch.add(a_last_out, a_api_out)
            # B = torch.add(b_last_out, b_api_out)
            A = torch.cat((a_last_out, a_api_out), dim=1)
            B = torch.cat((b_last_out, b_api_out), dim=1)


        else:
            A, B = self.get_i_embed(inputs[0], inputs[1], inputs[3], inputs[4])

        a_out = F.normalize(A, p=2, dim=1)
        b_out = F.normalize(B, p=2, dim=1)

        predictions = F.cosine_similarity(a_out, b_out, dim=1)

        return predictions,a_out,b_out




    def get_i_embed(self, g1_adj,g1_inputs,g2_adj,g2_inputs):
        a_embeds = self.i_embedding((g1_inputs.to(DEVICE))).to(DEVICE)
        # a_embeds = self.drop(a_embeds)

        b_embeds = self.i_embedding((g2_inputs.to(DEVICE))).to(DEVICE)
        # b_embeds = self.drop(b_embeds)

        return self.get_lstm_attention(g1_adj,a_embeds,g2_adj,b_embeds)

    def get_lstm_attention(self,g1_adj,g_a,g2_adj,g_b):

        a_out, (a_hidden, a_cell) = self.i_encoder(g_a)
        b_out, (b_hidden, b_cell) = self.i_encoder(g_b)
        # print('a_out', a_out.size())

        a_last_out = a_out
        b_last_out = b_out

        a_last_out = self.i_fc1(a_last_out)
        b_last_out = self.i_fc1(b_last_out)
        # print('a_last_out', a_last_out.size())

        g_a_node_matrix = self.i_fc3(self.attention_lstm(a_last_out))
        g_b_node_matrix = self.i_fc3(self.attention_lstm(b_last_out))
        # print('g_a_node_matrix',g_a_node_matrix.size())

        g_a_node_matrix = g_a_node_matrix.reshape(self.graph_num, -1, self.dense_dim)
        g_b_node_matrix = g_b_node_matrix.reshape(self.graph_num, -1, self.dense_dim)
        # print('g_a_node_matrix',g_a_node_matrix.size())

        # print('pairs[0]',pairs[0].shape)

        a_batch_graph = self.get_graph_embedding(g1_adj, g_a_node_matrix)
        b_batch_graph = self.get_graph_embedding(g2_adj, g_b_node_matrix)

        a_batch_graph = F.normalize(torch.stack(a_batch_graph), p=2, dim=1)
        b_batch_graph = F.normalize(torch.stack(b_batch_graph), p=2, dim=1)
        return  a_batch_graph ,b_batch_graph

    def get_kbadd_embed(self,g1_adj, a_kb,g2_adj,b_kb):
        a_api_embeds = self.kb_embedding((a_kb.to(DEVICE))).to(DEVICE)
        # a_api_embeds = self.drop(a_api_embeds)
        b_api_embeds = self.kb_embedding((b_kb.to(DEVICE))).to(DEVICE)
        # b_api_embeds = self.drop(b_api_embeds)
        return self.get_kb_attention(g1_adj,a_api_embeds,g2_adj,b_api_embeds)

    def get_kbcat_embed(self,a_kb,b_kb):
        a_api_embeds = self.kb_embedding((a_kb.to(DEVICE))).to(DEVICE)
        # a_api_embeds = self.drop(a_api_embeds)
        b_api_embeds = self.kb_embedding((b_kb.to(DEVICE))).to(DEVICE)
        # b_api_embeds = self.drop(b_api_embeds)
        return self.get_kbcat_attention(a_api_embeds,b_api_embeds)

    def get_kbcat_attention(self, a_api_embeds, b_api_embeds):
        a_api_out, (a_hidden, a_cell) = self.kb_encoder(a_api_embeds)
        b_api_out, (b_hidden, b_cell) = self.kb_encoder(b_api_embeds)

        a_api_out = self.kbcat_fc1(a_api_out)
        b_api_out = self.kbcat_fc1(b_api_out)

        a_api_out = self.kbcat_fc2(self.attention_kb(a_api_out))
        b_api_out = self.kbcat_fc2(self.attention_kb(b_api_out))


        return a_api_out, b_api_out
    def get_kb_attention(self,g1_adj,a_api_embeds,g2_adj,b_api_embeds):
        a_api_out, (a_hidden, a_cell) = self.kb_encoder(a_api_embeds)
        b_api_out, (b_hidden, b_cell) = self.kb_encoder(b_api_embeds)

        a_api_out = self.kb_fc1(a_api_out)
        b_api_out = self.kb_fc1(b_api_out)

        a_api_out = self.kb_fc3(self.attention_kb(a_api_out))
        b_api_out = self.kb_fc3(self.attention_kb(b_api_out))

        a_api_out = a_api_out.reshape(self.graph_num, -1, self.dense_dim)
        a_api_out = a_api_out.reshape(self.graph_num, -1, self.dense_dim)
        # print('g_a_node_matrix',g_a_node_matrix.size())

        # print('pairs[0]',pairs[0].shape)

        a_batch_graph = self.get_graph_embedding(g1_adj, a_api_out)
        b_batch_graph = self.get_graph_embedding(g2_adj, a_api_out)

        a_batch_graph = F.normalize(torch.stack(a_batch_graph), p=2, dim=1)
        b_batch_graph = F.normalize(torch.stack(b_batch_graph), p=2, dim=1)


        return a_batch_graph, b_batch_graph
    # without api or kb
    def get_attention_similarity(self,pairs):
        a_out, self.a_hidden = self.encoder(pairs[1], self.a_hidden)
        b_out, self.b_hidden = self.encoder(pairs[4], self.b_hidden)

        a_last_out = a_out
        b_last_out = b_out

        a_last_out = self.fc1(a_last_out)
        b_last_out = self.fc1(b_last_out)

        g_a_node_matrix = self.fc3(self.attention_lstm(a_last_out))
        g_b_node_matrix = self.fc3(self.attention_lstm(b_last_out))
        # print('g_a_node_matrix',g_a_node_matrix.size())
        g_a_node_matrix = g_a_node_matrix.reshape(self.graph_num, -1, self.dense_dim)
        g_b_node_matrix = g_b_node_matrix.reshape(self.graph_num, -1, self.dense_dim)
        # print('g_a_node_matrix',g_a_node_matrix.size())
        # print('pairs[0]',pairs[0].shape)

        a_batch_graph = self.get_graph_embedding(torch.Tensor(pairs[0]), g_a_node_matrix)
        b_batch_graph = self.get_graph_embedding(torch.Tensor(pairs[3]), g_b_node_matrix)

        cos_similarity = self.get_cos_similarity(a_batch_graph, b_batch_graph)

        # a_last_out = F.normalize(a_last_out, p=2, dim=1)
        # b_last_out = F.normalize(b_last_out, p=2, dim=1)
        return cos_similarity

    def get_lstm_similarity(self,pairs):
        a_out, self.a_hidden = self.encoder(pairs[1], self.a_hidden)
        b_out, self.b_hidden = self.encoder(pairs[4], self.b_hidden)

        # a_last_out = self.fc(self.a_hidden[0])
        # b_last_out = self.fc(self.b_hidden[0])
        a_last_out = self.fc1(a_out[:,-1,:])
        b_last_out = self.fc1(b_out[:,-1,:])

        A = a_last_out.size()
        # print(A)
        # a_block_matrix = a_last_out.reshape((A[1], A[2])).to(DEVICE)
        a_block_matrix = a_last_out.to(DEVICE)
        # block_matrix = block_matrix.cpu().detach().numpy()
        g_a_node_matrix = a_block_matrix.reshape(self.graph_num, -1, A[1])
        # print('g_a_node_matrix \n', g_a_node_matrix)
        # g_a_node_matrix = F.normalize(g_a_node_matrix,p=2,dim=1)
        # print('g_a_node_matrix.size() \n', g_a_node_matrix.size())
        # print('g_a_node_matrix \n', g_a_node_matrix)

        B = b_last_out.size()
        # print("B",B)
        # b_block_matrix = b_last_out.reshape((B[1], B[2])).to(DEVICE)
        b_block_matrix = b_last_out.to(DEVICE)
        # print('b_block_matrix \n', b_block_matrix.size())
        # block_matrix = block_matrix.cpu().detach().numpy()
        # g_b_node_matrix = b_block_matrix.reshape(self.graph_num, -1, b_block_matrix.shape[1])
        g_b_node_matrix = b_block_matrix.reshape(self.graph_num, -1, B[1])
        # g_a_node_matrix = F.normalize(g_b_node_matrix, p=2, dim=1)


        a_batch_graph = self.get_graph_embedding(torch.Tensor(pairs[0]), g_a_node_matrix)
        b_batch_graph = self.get_graph_embedding(torch.Tensor(pairs[3]), g_b_node_matrix)

        cos_similarity = self.get_cos_similarity(a_batch_graph,b_batch_graph)
        # cos_similarity = self.get_graph_similarity(a_batch_graph, b_batch_graph)

        return cos_similarity

    def get_cos_similarity(self,A,B):
        a_graph = F.normalize(torch.stack(A),p=2,dim=1)
        b_graph = F.normalize(torch.stack(B), p=2, dim=1)
        similarity = F.cosine_similarity(a_graph,b_graph,dim=1)
        # print(similarity)
        return similarity

    def matrix_load(self,id2vec):
        matrix = np.load(id2vec)
        return matrix

    def get_graph_embedding(self,adj,node):
        g_embedding = []
        # print("get_graph_embedding")
        for i in range(0,len(adj)):
            g_embedding.append(self.get_one_graph_embedding(adj[i].to(DEVICE),node[i]).to(DEVICE))
        return g_embedding

    def get_one_graph_embedding(self,adj,node):
        # print("get_one_graph_embedding")
        node_init = torch.matmul(node,self.w1).to(DEVICE)
        adj = torch.as_tensor(adj,dtype=torch.float)
        graph = torch.matmul(adj,node_init).to(DEVICE)
        A = node_init.to(DEVICE)

        for i in range(0,self.graph_iteration-1):
            B = graph.to(DEVICE)
            node_iterration = self.node_iteration-1
            while node_iterration >= 0:
                node_updata = torch.matmul(B,self.w_n[node_iterration])
                # node_updata = np.matmul(B, w_n[0])
                if node_iterration > 0 :
                    B = F.leaky_relu(node_updata)
                else:
                    B = node_updata
                iter = node_iterration -1
                node_iterration = iter

            A = torch.tanh(A+B).to(DEVICE)
            graph = torch.matmul(adj,A).to(DEVICE)

        t = torch.sum(A,dim=1).to(DEVICE)
        # print('np.sum(A)',t)
        # print('w2',w2.shape)

        graph_embedding = torch.matmul(t,self.w2).to(DEVICE)
        graph_embedding = self.fc2(graph_embedding)
        # print("graph_embedding",graph_embedding.shape)
        return graph_embedding


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--word2id", type=str, help="word2id to analyze")
    parser.add_argument("-v", "--id2vec", type=str, help="id2vec to analyze")
    parser.add_argument("-d", "--dataset", type=str, help="dir for analyze")
    args = parser.parse_args()


    print(time.asctime(time.localtime(time.time())))

    a_data = [45,345,3456,89,2533]
    b_data = [45, 895, 6, 89, 2533]
    c_data = [45, 345, 3456, 89, 209]
    data= []

    f = np.asarray(a_data[0:50])
    length = f.shape[0]
    if f.shape[0] < 50:
        f = np.pad(f, (0, 50 - f.shape[0]), mode='constant')

    x_data = torch.from_numpy(f)
    # print(x_data.size())
    embedding_dim, hidden_dim, num_layers = 100,100,1
    lstm = LSTMNetwork(args.id2vec,embedding_dim, hidden_dim, num_layers)
    # init parameter
    lstm.zero_grad()
    lstm.hidden = lstm.init_hidden()

    out = lstm(x_data)

    print(out)
    print(out.size())



    print(time.asctime(time.localtime(time.time())))