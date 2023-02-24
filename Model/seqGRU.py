import torch
from torch import nn
import torch.nn.functional as F

torch.manual_seed(1234)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class seqGRUNetwork(nn.Module):

    def __init__(self, id2vec, kb2vec, output_dim, embedding_dim, hidden_dim, num_layers, max_nodes, batch_size,
                 with_kb):
        super(seqGRUNetwork, self).__init__()
        self.output_dim = output_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.max_nodes = max_nodes

        self.with_kb = with_kb

        self.id2vec = id2vec
        self.kb2vec = kb2vec
        self.dense_dim = 2000

        self.seqGRU = nn.GRU(input_size=self.embedding_dim,
                             hidden_size=self.hidden_dim,
                             num_layers=self.num_layers,
                             bidirectional=True,
                             batch_first=False).to(DEVICE)

        self.fc1 = nn.Sequential(nn.Linear(self.hidden_dim*2, self.dense_dim, bias=True))
        self.fc2 = nn.Sequential(nn.Linear(self.dense_dim, self.output_dim, bias=True))

        self.kb_seqGRU = nn.GRU(input_size=self.embedding_dim,
                                hidden_size=self.hidden_dim,
                                num_layers=self.num_layers,
                                bidirectional=True,
                                batch_first=False).to(DEVICE)

        # self.kb_fc = nn.Sequential(nn.Linear(self.hidden_dim, self.output_dim, bias=True))

        self.drop = nn.Dropout(0)

        # word2vec for task-related
        # self.i_embedding =  nn.Embedding(self.id2vec.shape[0], embedding_dim)
        # self.kb_embedding = nn.Embedding(self.kb2vec.shape[0], embedding_dim)

        #x2v for task independent
        self.i_embedding = nn.Embedding(self.id2vec.shape[0], embedding_dim)
        self.i_embedding.weight = nn.Parameter(torch.tensor(self.id2vec,dtype=torch.float32))
        self.i_embedding.weight.requires_grad = False

        # k2v
        self.kb_embedding = nn.Embedding(self.kb2vec.shape[0], embedding_dim)
        self.kb_embedding.weight = nn.Parameter(torch.tensor(self.kb2vec,dtype=torch.float32))
        self.kb_embedding.weight.requires_grad = False

        self.w_omega = nn.Parameter(torch.Tensor(self.dense_dim, self.dense_dim))
        self.u_omega = nn.Parameter(torch.Tensor(self.dense_dim, 1))

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

    def forward(self, inputs):
        if self.with_kb:
            a_last_out, b_last_out = self.get_i_embed(inputs)
            a_api_out, b_api_out = self.get_kb_embed(inputs)

            # a_last_out = self.fc(a_last_out)
            # b_last_out = self.fc(b_last_out)
            # a_api_out = self.kb_fc(a_api_out)
            # b_api_out = self.kb_fc(b_api_out)

            A = torch.cat((a_last_out, a_api_out), dim=1)
            B = torch.cat((b_last_out, b_api_out), dim=1)

            # A = torch.add(a_last_out, a_api_out)
            # B = torch.add(b_last_out, b_api_out)
        else:
            A, B = self.get_i_embed(inputs)

            # A = self.fc(a_last_out)
            # B = self.fc(b_last_out)

        a_last_out = F.normalize(A, p=2, dim=1)
        b_last_out = F.normalize(B, p=2, dim=1)

        predictions = F.cosine_similarity(a_last_out, b_last_out, dim=1)

        return predictions,a_last_out,b_last_out

    def get_i_embed(self, inputs):
        a_embeds = self.i_embedding((inputs[0].to(DEVICE))).to(DEVICE)
        a_embeds = self.drop(a_embeds)
        b_embeds = self.i_embedding((inputs[1].to(DEVICE))).to(DEVICE)
        b_embeds = self.drop(b_embeds)
        a_out, a_hidden= self.seqGRU(a_embeds)
        b_out, b_hidden = self.seqGRU(b_embeds)

        a_last_out = a_out
        b_last_out = b_out

        a_last_out = self.fc1(a_last_out)
        b_last_out = self.fc1(b_last_out)

        a_last_out = self.fc2(self.attention_GRU(a_last_out))
        b_last_out = self.fc2(self.attention_GRU(b_last_out))

        # a_last_out = a_out[:, -1, :]
        # b_last_out = b_out[:, -1, :]
        return a_last_out, b_last_out

    def get_kb_embed(self, inputs):
        a_api_embeds = self.kb_embedding((inputs[3].to(DEVICE))).to(DEVICE)
        a_api_embeds = self.drop(a_api_embeds)
        b_api_embeds = self.kb_embedding((inputs[4].to(DEVICE))).to(DEVICE)
        b_api_embeds = self.drop(b_api_embeds)

        a_api_out, (a_hidden, a_cell) = self.kb_seqGRU(a_api_embeds)
        b_api_out, (b_hidden, b_cell) = self.kb_seqGRU(b_api_embeds)

        # a_api_out = a_api_out[:, -1, :]
        # b_api_out = b_api_out[:, -1, :]

        a_api_out = self.fc1(a_api_out)
        b_api_out = self.fc1(b_api_out)

        a_api_out = self.fc2(self.attention_GRU(a_api_out))
        b_api_out = self.fc2(self.attention_GRU(b_api_out))

        return a_api_out, b_api_out

    def attention_GRU(self, x):

        u = torch.tanh(torch.matmul(x, self.w_omega))
        att = torch.matmul(u, self.u_omega)

        att_score = F.softmax(att, dim=1)
        # print(att_score.size())
        scored_x = x * att_score
        # print(scored_x.size())
        att_out = torch.sum(scored_x, dim=1)
        # print(att_out.size())
        return att_out




