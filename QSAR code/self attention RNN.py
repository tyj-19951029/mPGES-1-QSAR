# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader
from torch.autograd import Variable
from collections import Counter
from rdkit import Chem
import re
import sklearn
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
#pre training
mols_suppl = Chem.SDMolSupplier(r"D:\academic\degree_doctor\project\mPGES\QSAR\RNN\drugbank_structures.sdf")
smi_list = []
char_list = []
metal_pattern = r"Li|Na|K|Rb|Cs|Fr|Be|Mg|Ca|Sr|Ba|Ra|Sc|Y|La|Ac|Ti|Zr|Hf|V|Nb|Ta|Cr|Mo|W|Mn|Tc|Re|Fe|Ru|Os|Co|Rh|lr|Ni|Pd|Pt|Cu|Ag|Au|Zn|Cd|Hg|Al|Ga|Ge|ln|Sn|Sb|Tl|Pb|Bi|Po|Ce|Pr|Nd|Pm|Sm|Eu|Gd|Tb|Dy|Ho|Er|Tm|Yb|Lu|Th|Pa|U|Np|Pu|Am|Cm|Bk|Cf|Es|Fm|Md|No|Lr|Rf|Ha|Sg|Bh|Hs|Mt"
metal_valence = "M\+\w|M\+|M\w+"
non_metal = "Si|As|Te|At|Se|se"
nobel_gas = "He|Ne|Ar|Kr|Xe|Rn"

for i in range(len(mols_suppl)):
    mol = mols_suppl[i]
    try:
        smiles = Chem.MolToSmiles(mol)
        smiles = smiles.replace('Cl','Q')#Q for Cl
        smiles = smiles.replace('Br','W')#W for Br
        smiles = re.sub(metal_pattern,'M',smiles)
        smiles = re.sub(metal_valence,'M',smiles)#M for Mg+2
        smiles = re.sub(non_metal,'X',smiles)#substitute X for the nonmetal
        smiles = re.sub(nobel_gas,'T',smiles)#substitute T for the noble gases
        if len(smiles) >= 10:
            smi_list.append(smiles)
            for c in smiles:
                char_list.append(c) 
    except:
        print(i," can not transform to SMILES")

char_dict = dict(Counter(char_list))

sum_ = 0
for smi in smi_list:
    sum_ += len(smi)
mean_len = sum_//len(smi_list)

SMILES_counts = np.array([char for char in char_dict.values()], dtype=np.float32)

chars = set()
for smi in smi_list:
    chars = chars.union(set(char for char in smi))
chars = sorted(list(chars))
#Characters correspond to numbers
smiles2index = dict((char, i+1) for i,char in enumerate(chars))
smiles2index['<UNK>'] = len(smiles2index)+1#UNK is the sum of all unknown characters
#Numeric correspondence character
index2smiles = dict((i+1,char) for i,char in enumerate(chars))
word_freqs = SMILES_counts / np.sum(SMILES_counts)#The frequency at which each character appears
word_freqs = word_freqs ** (3./4.)#This value of 0.75 is recommended by the original literature


class WordEmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, char_list, smiles2index, index2smiles, word_freqs, SMILES_counts,context_word,noise):
        super(WordEmbeddingDataset, self).__init__() # Initialize the model through the parent class, then override the two methods
        self.text_encoded = [smiles2index.get(word, smiles2index['<UNK>']) for word in char_list] # To digitize a word that is not in the dictionary, return the index corresponding to UNK
        self.text_encoded = torch.LongTensor(self.text_encoded) 
        self.index2smiles = index2smiles
        self.word_freqs = torch.Tensor(word_freqs)
        self.SMILES_counts = torch.Tensor(SMILES_counts)
        self.context_word = context_word
        self.noise = noise
        
        
    def __len__(self):
        return len(self.text_encoded) # Returns the total number of words, the total number of items
    
    def __getitem__(self, idx):
        ''' 这个function返回以下数据用于训练
            - 中心词
            - 这个单词附近的positive word
            - 随机采样的K个单词作为negative word
        '''
        center_words = self.text_encoded[idx] # Get the central word
        pos_indices = list(range(idx - self.context_word, idx)) + list(range(idx + 1, idx + self.context_word + 1)) 
        pos_indices = [i % len(self.text_encoded) for i in pos_indices] 
        pos_words = self.text_encoded[pos_indices] # tensor(list)
        
        neg_words = torch.multinomial(self.word_freqs, self.noise * pos_words.shape[0], True)
# The sampling method adopts the sampling with the put back, and the higher the self.word_freqs value, the higher the sampling probability
        
        return center_words, pos_words, neg_words

context_word = 3
noise = 15
dataset = WordEmbeddingDataset(char_list, smiles2index, index2smiles, word_freqs, SMILES_counts,context_word,noise)
dataloader = torch.utils.data.DataLoader(dataset, batch_size = 256, shuffle=True, num_workers = 0)


#Embedding
class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(EmbeddingModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        
        self.in_embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.out_embed = nn.Embedding(self.vocab_size, self.embed_size)
        
    def forward(self, input_labels, pos_labels, neg_labels):
        ''' input_labels: center words, [batch_size]
            pos_labels: positive words, [batch_size, (window_size * 2)]
            neg_labels：negative words, [batch_size, (window_size * 2 * K)]
            
            return: loss, [batch_size]
        '''
        input_embedding = self.in_embed(input_labels) # [batch_size, embed_size]
        pos_embedding = self.out_embed(pos_labels)# [batch_size, (window * 2), embed_size]
        neg_embedding = self.out_embed(neg_labels) # [batch_size, (window * 2 * K), embed_size]
        
        input_embedding = input_embedding.unsqueeze(2) # [batch_size, embed_size, 1]
        
        pos_dot = torch.bmm(pos_embedding, input_embedding) # [batch_size, (window * 2), 1]
        pos_dot = pos_dot.squeeze(2) # [batch_size, (window * 2)]
        
        neg_dot = torch.bmm(neg_embedding, -input_embedding) # [batch_size, (window * 2 * K), 1]
        neg_dot = neg_dot.squeeze(2) # batch_size, (window * 2 * K)]
        
        log_pos = F.logsigmoid(pos_dot).sum(1) 
        log_neg = F.logsigmoid(neg_dot).sum(1)
        
        loss = log_pos + log_neg
        
        return -loss
    
    def input_embedding(self):
        return self.in_embed.weight
    
#training model
vocab_size = len(smiles2index)
embed_size = 2
model = EmbeddingModel(vocab_size,embed_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0005)

epochs =20
loss_list = []
for e in range(epochs):
    for i, (input_labels, pos_labels, neg_labels) in enumerate(dataloader):
        input_labels = input_labels.long().to(device)
        pos_labels = pos_labels.long().to(device)
        neg_labels = neg_labels.long().to(device)

        optimizer.zero_grad()
        loss = model(input_labels, pos_labels, neg_labels).mean()
        loss_list.append(loss.item())
        loss.backward()

        optimizer.step()

        if i % 500 == 0:
            print('epoch', e, 'iteration', i, loss.item())
    PATH = r"D:\academic\degree_doctor\project\mPGES\QSAR\RNN\hide\word2vec_logs"+'\\'+str(e)+r"_drugbank.pth"
    torch.save(model, PATH)
print("finsih training SMILES2VEC")

#Loss function diagram of the model
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family']='Times New Roman'
plt.figure(figsize = (25,15))
train_times = np.linspace(0, len(loss_list)-1, 1000, dtype=int)
loss_ = []
for i in train_times:
    loss_.append(loss_list[i])
plt.plot(train_times, loss_, color='red', label = 'loss')
plt.ylabel('loss',fontsize=35)
plt.xlabel('training times',fontsize=35)
plt.title('SMILES2vec loss',fontsize=50)
plt.legend(fontsize=35)
plt.savefig(r"D:\academic\degree_doctor\project\mPGES\RNN\loss_smiles2vec3.tif",dpi=300,bbox_inches='tight')
plt.show()
#In order to characterize the distribution of SMILES training results for DRUGBANK data, 
#poor training is indicated if the distribution between each atom is scattered and not different
import matplotlib
matplotlib.rcParams['font.family']='Times New Roman'
plt.figure(figsize = (20,20))
x_scatter = []
y_scatter = []
for i in range(len(embedding)):
    x_scatter.append(embedding[i][0])
    y_scatter.append(embedding[i][1])
c=list(range(len(embedding)))
c5 = [5*i for i in c]
plt.scatter(x_scatter, y_scatter, s=c5, c=c, marker='o')
plt.title('SMILES2vec',fontsize=25)
plt.colorbar(label='smiles2index')
plt.savefig(r"D:\academic\degree_doctor\project\mPGES\RNN\smiles2vec3.png",dpi =300,bbox_inches='tight')
plt.show()



#To load the pre-trained model, run the class EmbeddingModel section again
model_1 = torch.load(r"D:\academic\degree_doctor\project\mPGES\QSAR\RNN\hide\word2vec_logs\12_drugbank.pth")
print(model_1)
embedding_weights_1 = model_1.input_embedding().cpu()
embedding_1 = embedding_weights_1.detach().numpy()


atom_list = []
data_csv = pd.read_csv(r"D:\academic\degree_doctor\project\mPGES\QSAR\RNN\data_735.csv")

metal_pattern = r"Li|Na|K|Rb|Cs|Fr|Be|Mg|Ca|Sr|Ba|Ra|Sc|Y|La|Ac|Ti|Zr|Hf|V|Nb|Ta|Cr|Mo|W|Mn|Tc|Re|Fe|Ru|Os|Co|Rh|lr|Ni|Pd|Pt|Cu|Ag|Au|Zn|Cd|Hg|Al|Ga|Ge|ln|Sn|Sb|Tl|Pb|Bi|Po|Ce|Pr|Nd|Pm|Sm|Eu|Gd|Tb|Dy|Ho|Er|Tm|Yb|Lu|Th|Pa|U|Np|Pu|Am|Cm|Bk|Cf|Es|Fm|Md|No|Lr|Rf|Ha|Sg|Bh|Hs|Mt"
metal_valence = "M\+\w|M\+|M\w+"
non_metal = "Si|As|Te|At|Se|se"
nobel_gas = "He|Ne|Ar|Kr|Xe|Rn"
smile_list = []
labels = []
for i in range(len(data_csv)):
    smiles = data_csv['SMILES'][i]
    try:
        smiles = smiles.replace('Cl','Q')
        smiles = smiles.replace('Br','W')
        smiles = re.sub(metal_pattern,'M',smiles)
        smiles = re.sub(metal_valence,'M',smiles)
        smiles = re.sub(non_metal,'X',smiles)
        smiles = re.sub(nobel_gas,'T',smiles)
        smile_list.append(smiles)
        atom_list.append([ c for c in smiles] )
        labels.append(data_csv['label'][i])
    except:
        print(i," can not transform to SMILES")
#The character corresponds to the embedding result
smiles2embedding = []
for i in range(len(embedding_1)):
    smiles2embedding.append((list(smiles2index.keys())[i],embedding_1[i]))
smiles2embedding = dict(smiles2embedding)
chars_size=len(smiles2embedding)
max_length = max(len(s) for s in smile_list)

#ATTENTION RNN
def encode(smiles_list,smiles_dict):
    length = []

    out_smiles = [[smiles_dict.get(c,smiles2index['<UNK>']) for c in smiles] for smiles in smiles_list]
    length = [len(smiles) for smiles in smiles_list]

    return out_smiles,length

X,len_X  = encode(smile_list, smiles2embedding)
Y_data = np.array(labels)

def get_minibatches(n, minibatch_size, shuffle = False):#If shuffle = true shuffles the order between batches and n is the total number
    idx_list = np.arange(0, n, minibatch_size)
    if shuffle:
        np.random.shuffle(idx_list)
    minibatches = []
    for index in idx_list:
        minibatches.append(np.arange(index, min(index + minibatch_size, n)))
    return minibatches

def prepare_data(smiles_list,embedd_size):
    lengths = [len(smile) for smile in smiles_list]
    n_sample = len(smiles_list)
    max_len = np.max(lengths)
    
    x = np.zeros((n_sample, max_len,embedd_size)).astype('float32')
    x_len = np.array(lengths).astype('float32')
    for index, smile in enumerate(smiles_list):
        x[index, :lengths[index]] = smile
    return x, x_len

def gen_examples(smiles, batch_size,embedd_size):
    minibatches = get_minibatches(len(smiles), batch_size)
    all_ex = []
    all_ex_len = []
    for minibatch in minibatches:
        mb_smiles = [smiles[i] for i in minibatch]    

        mb_x, mb_x_len = prepare_data(mb_smiles,embedd_size)
        
        all_ex_len.append(mb_x_len)
        all_ex.append((mb_x))
    return all_ex,all_ex_len

X_data,seq_len_list = gen_examples(X, len(X),embedd_size = 2)
seq_len_list = seq_len_list[0]
MAX_LENGTH = int(max(seq_len_list))
np.array(X_data).shape


#calculate the attention weight of each compound
for x_array in X_data:
    x_all = torch.from_numpy(x_array)
    label = np.array(Y_data)
    label = torch.from_numpy(label)
    trainset = torch.utils.data.TensorDataset(x_all,label)
    ALLloader = torch.utils.data.DataLoader(trainset, batch_size = 1, shuffle=False, num_workers=0)
    
#The training set and test set were divided into batch, with one compound in each batch
for x_array in X_data:
    X_train, X_test, y_train_0, y_test_0 = train_test_split( x_array ,Y_data, test_size = 0.25, random_state = 42)
    X_train_1 = torch.from_numpy(X_train)
    X_test_1 = torch.from_numpy(X_test)
    y_train = torch.from_numpy(y_train_0)
    y_test = torch.from_numpy(y_test_0)
    trainset = torch.utils.data.TensorDataset(X_train_1,y_train)
    testset = torch.utils.data.TensorDataset(X_test_1,y_test)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size= 1, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size= 1, shuffle=True, num_workers=0)
    
class GRU_attention(nn.Module):
    def __init__(self,hidden_size ,vocab_size , embed_size,classes):
        super(GRU_attention, self).__init__()
        self.hidden_dim = hidden_size

        # To match the dimensions of the Attention operation, hidden_DIM is even
        self.bigru = nn.GRU(embed_size, self.hidden_dim // 2, num_layers=2, bidirectional=True)
        self.weight_W = nn.Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))
        self.weight_proj = nn.Parameter(torch.Tensor(self.hidden_dim, 1))
        self.fc = nn.Linear(self.hidden_dim, classes)

        nn.init.uniform_(self.weight_W, -0.1, 0.1)
        nn.init.uniform_(self.weight_proj, -0.1, 0.1)

    def forward(self, inputs):

        embeds = inputs.view(1,-1,embed_size) # [seq_len, bs, emb_dim]
        
        embeds = embeds.permute(1,0,2)

        gru_out, _ = self.bigru(embeds) # [seq_len, bs, hid_dim]
        x = gru_out.permute(1, 0, 2) # Change the dimension
        # # # Attention
        u = torch.tanh(torch.matmul(x, self.weight_W))
        att = torch.matmul(u, self.weight_proj)
        att_score = F.softmax(att, dim=1)
        scored_x = x * att_score
        # # # Attention finish
        feat = torch.sum(scored_x, dim=1) 
        y = self.fc(feat)
        out = y
        return out,att_score
        
classes = 1
vocab_size = (chars_size + 1)
hidden_size = 102#It needs to be even
output_size = 2
embed_size=2
AttRNN = GRU_attention(hidden_size,vocab_size,embed_size,classes).to(device)
learning_rate = 0.005
epoch = 200
optimizer = torch.optim.Adam(AttRNN.parameters(), lr = learning_rate)
#criterion = nn.BCELoss()
criterion = nn.SmoothL1Loss()
train_loss_list = []
test_loss_list = []
        
###RNN regression
for m in range(epoch):
    training_loss = 0.0
    train_true_ = []
    train_pred_ = []
    for index, data in enumerate(trainloader):
        AttRNN.train()
        AttRNN.zero_grad()
        inputs, labels = data
        inputs = Variable(inputs.view(-1, 1, MAX_LENGTH)).to(device)
        labels = Variable(labels).float().to(device)
        optimizer.zero_grad()      
        output,att_weight = AttRNN(inputs)
        train_true_.append(np.array(labels.cpu().detach()).item())
        train_pred_.append(np.array(output.cpu().detach()).item())
        output = output.view(1)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        training_loss += loss.item()
       
    m_train_loss_mean = training_loss/len(X_train)
    train_loss_list.append(m_train_loss_mean)
    PATH = r"D:\academic\degree_doctor\project\mPGES\QSAR\RNN\hide103\smiles2vec_rnn_logs"+"\\"+str(m)+r".pth"
    torch.save(AttRNN, PATH)
    with torch.no_grad():
        AttRNN.eval()
        total_test = 0
        correct_test = 0
        total_train = 0
        correct_train = 0
        test_true = []
        test_pred = []
        train_true = []
        train_pred = []
        for index, data in enumerate(testloader):
             inputs, labels = data
             labels = Variable(labels).float().to(device)
             inputs = Variable(inputs.view(-1, 1, MAX_LENGTH)).to(device)
             output,att_weight = AttRNN(inputs)
            
             test_true.append(np.array(labels.cpu()).item())
             test_pred.append(np.array(output.cpu()).item())
        for index, data in enumerate(trainloader):
             inputs, labels = data
             labels = Variable(labels).float().to(device)
             inputs = Variable(inputs.view(-1, 1, MAX_LENGTH)).to(device)
             output,att_weight = AttRNN(inputs)
             
             train_true.append(np.array(labels.cpu()).item())
             train_pred.append(np.array(output.cpu()).item())
             
    print(m, 'epochs test r2:', sklearn.metrics.r2_score(test_true,test_pred),"train r2:",sklearn.metrics.r2_score(train_true,train_pred))
print("finish",m,"epoch")


#trainng again
model_2 = torch.load(r"D:\academic\degree_doctor\project\mPGES\QSAR\RNN\hide103\smiles2vec_rnn_logs\170.pth").to(device)
optimizer_1 = torch.optim.Adam(model_2.parameters(), lr=0.0001)
#Optimizer_1 name Remember the corresponding, as well as the name of the imported model to optimize on
epoch_1 = 200
for m in range(epoch_1):
    training_loss = 0.0
    train_true_ = []
    train_pred_ = []
    for index, data in enumerate(trainloader):
        model_2.train()
        model_2.zero_grad()
        inputs, labels = data
        inputs = Variable(inputs.view(-1, 1, MAX_LENGTH)).to(device)
        labels = Variable(labels).float().to(device)
        optimizer_1.zero_grad()      
        output,att_weight = model_2(inputs)
        train_true_.append(np.array(labels.cpu().detach()).item())
        train_pred_.append(np.array(output.cpu().detach()).item())
        loss = criterion(output, labels)
        loss.backward()
        optimizer_1.step()
        training_loss += loss.item()

    m_train_loss_mean = training_loss/len(X_train)
    train_loss_list.append(m_train_loss_mean)
    print(m,"epoch train loss:",m_train_loss_mean)
    PATH = r"D:\academic\degree_doctor\project\mPGES\QSAR\RNN\hide103\smiles2vec_rnn_logs"+"\\170_"+str(m)+r".pth"
    torch.save(model_2, PATH)
    with torch.no_grad():
        model_2.eval()
        total_test = 0
        correct_test = 0
        total_train = 0
        correct_train = 0
        test_true = []
        test_pred = []
        train_true = []
        train_pred = []
        for index, data in enumerate(testloader):
             inputs, labels = data
             labels = Variable(labels).float().to(device)
             inputs = Variable(inputs.view(-1, 1, MAX_LENGTH)).to(device)
             output,att_weight = model_2(inputs)
            
             test_true.append(np.array(labels.cpu()).item())
             test_pred.append(np.array(output.cpu()).item())
        for index, data in enumerate(trainloader):
             inputs, labels = data
             labels = Variable(labels).float().to(device)
             inputs = Variable(inputs.view(-1, 1, MAX_LENGTH)).to(device)
             output,att_weight = model_2(inputs)
             
             train_true.append(np.array(labels.cpu()).item())
             train_pred.append(np.array(output.cpu()).item())
             
    print(m,'te r2:', sklearn.metrics.r2_score(test_true,test_pred),"tr r2:",sklearn.metrics.r2_score(train_true,train_pred))
    print(m,'te MSE:', sklearn.metrics.mean_squared_error(test_true,test_pred),"tr MSE:",sklearn.metrics.mean_squared_error(train_true,train_pred))
print("finish",m,"epoch") 

#repeat training
model_2 = torch.load(r"D:\academic\degree_doctor\project\mPGES\QSAR\RNN\smiles2vec_rnn_logs\47_96.pth").to(device)#选择上一轮训练中效果最好的模型
optimizer_1 = torch.optim.Adam(model_2.parameters(), lr=0.00005)
epoch_1 = 200
for m in range(epoch_1):
    training_loss = 0.0
    train_true_ = []
    train_pred_ = []
    for index, data in enumerate(trainloader):
        AttRNN.train()
        AttRNN.zero_grad()
        inputs, labels = data
        inputs = Variable(inputs.view(-1, 1, MAX_LENGTH)).to(device)
        labels = Variable(labels).float().to(device)
        optimizer.zero_grad()      
        output,att_weight = AttRNN(inputs)
        train_true_.append(np.array(labels.cpu().detach()).item())
        train_pred_.append(np.array(output.cpu().detach()).item())
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        training_loss += loss.item()

    m_train_loss_mean = training_loss/len(X_train)
    train_loss_list.append(m_train_loss_mean)
    print(m,"epoch train loss:",m_train_loss_mean)
    PATH = r"D:\academic\degree_doctor\project\mPGES\QSAR\RNN\smiles2vec_rnn_logs"+"\\47_96_"+str(m)+r".pth"
    torch.save(model_2, PATH)
    with torch.no_grad():
        AttRNN.eval()
        total_test = 0
        correct_test = 0
        total_train = 0
        correct_train = 0
        test_true = []
        test_pred = []
        train_true = []
        train_pred = []
        for index, data in enumerate(testloader):
             inputs, labels = data
             labels = Variable(labels).float().to(device)
             inputs = Variable(inputs.view(-1, 1, MAX_LENGTH)).to(device)
             output,att_weight = AttRNN(inputs)
            
             test_true.append(np.array(labels.cpu()).item())
             test_pred.append(np.array(output.cpu()).item())
        for index, data in enumerate(trainloader):
             inputs, labels = data
             labels = Variable(labels).float().to(device)
             inputs = Variable(inputs.view(-1, 1, MAX_LENGTH)).to(device)
             output,att_weight = AttRNN(inputs)
             
             train_true.append(np.array(labels.cpu()).item())
             train_pred.append(np.array(output.cpu()).item())
             
    print(m,'te r2:', sklearn.metrics.r2_score(test_true,test_pred),"tr r2:",sklearn.metrics.r2_score(train_true,train_pred))
    print(m,'te MSE:', sklearn.metrics.mean_squared_error(test_true,test_pred),"tr MSE:",sklearn.metrics.mean_squared_error(train_true,train_pred))
print("finish",m,"epoch") 


model_3 = torch.load(r"D:\academic\degree_doctor\project\mPGES\QSAR\RNN\smiles2vec_rnn_logs\47_96.pth").to(device)
all_smile_char_path = r"D:\academic\degree_doctor\project\mPGES\QSAR\RNN\hide103\old_weight" #Attention for all characters, including numbers and punctuation marks
only_atom_path = r"D:\academic\degree_doctor\project\mPGES\QSAR\RNN\hide103\new_weight"  ##The attention of all atoms
latent_list = []
with torch.no_grad():
    for index, data in enumerate(ALLloader):      
        inputs, labels = data
        inputs = Variable(inputs.view(-1, 1, MAX_LENGTH)).to(device)
        output,att_weight = model_3(inputs)
        att_weight = att_weight.cpu()
        att_weight = att_weight.detach().numpy().tolist()
        for x in att_weight:
            weight = x[:int(seq_len_list[index])]
            latent_list.append(weight)
for i in range(len(latent_list)):
    df = pd.DataFrame({'smiles':atom_list[i],'weight':latent_list[i]})
    path = all_smile_char_path +'\\'+str(i)+'_all_weight.csv' 
    df.to_csv(path)   

dele = ["#","%","+","(",")","-",".","/","0","1","2","3","4","5","6","7","8","9","=","@","[","\\","]","H"]
num_del = []
for i in range(735):
    path = all_smile_char_path +'\\'+str(i)+'_all_weight.csv'
    data = pd.read_csv(path)
    num_del.clear()
    for m in range(len(data)):
        if data["smiles"][m] in dele:
            num_del.append(m)
    data_new = data.drop(num_del)
    data_new.columns = ['ID','smiles','IC50']
    data_new = data_new.drop(['ID'],axis = 1)
    data_new.index = range(len(data_new))
    data_new.to_csv(only_atom_path +'\\'+str(i)+'_atom_weight.csv')
    
    
####Visualize the important atoms
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import PandasTools
import pandas as pd
data_csv= pd.read_csv(r"D:\academic\degree_doctor\project\mPGES\QSAR\RNN\data_735.csv")
Chem.PandasTools.AddMoleculeColumnToFrame(data_csv, smilesCol='SMILES', molCol='MOL', includeFingerprints=True)
opts = Draw.DrawingOptions()
opts.elemDict = {1: (0, 0, 0),
 7: (0, 0, 0),
 8: (0, 0, 0),
 9: (0, 0, 0),
 15: (0, 0, 0),
 16: (0, 0, 0),
 17: (0, 0, 0),
 35: (0, 0, 0),
 53: (0, 0, 0),
 0: (0, 0, 0)}

attention_draw_path = r"D:\academic\degree_doctor\project\mPGES\QSAR\RNN\hide103\weightmol"

for i in range(len(data_csv)):
    if data_csv['label'][i] >= 8:
        #The attention weight of each compound
        path = r"D:\academic\degree_doctor\project\mPGES\QSAR\RNN\hide103\new_weight" + '\\'+str(i)+'_atom_weight.csv'  
        data = pd.read_csv(path)
        weight = []
        for m in range(len(data)):
            str_weight = data['IC50'][m]
            num_weight = str_weight.strip('[')
            num_weight = float(num_weight.strip(']'))
            weight.append(num_weight)
        index_ = sorted(range(len(weight)), key=lambda x: weight[x], reverse = True)
        mol = data_csv['MOL'][i]
        img = Draw.MolToImage(mol,size=(800, 800), options = opts, highlightAtoms = index_[:len(index_)//3],highlightColor = [255,0,0])
        #img.show()
        path_smile = attention_draw_path + '\\' +str(i) +'.png'
        img.save(path_smile)