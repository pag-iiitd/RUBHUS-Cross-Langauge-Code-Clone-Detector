from torch.nn import Sequential, Linear, ReLU, GRU
from torch_geometric.data import Dataset, Data, DataLoader  
# from torch_geometric.datasets import QM9
from torch_geometric.nn import NNConv, Set2Set
from torch.nn import BCELoss, BCEWithLogitsLoss
from torch_geometric.utils import remove_self_loops
import numpy as np
import os
import os.path as osp
import random
import sys
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import time, itertools
from torch_geometric.utils import degree
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from torch.utils.tensorboard import SummaryWriter


#returns the encoding and adjacency matrix given the program filename
def graphToMatrices(filename):
    # move to global to i`mprove performance
    from matrixFormation import oneHotEncoder,adjacencyMatrixCreator
    languageType=""
    astOfProgram=[]
    if filename.split(".")[1] == "py":
        languageType = "python"
        listOfFiles =  open('python-asts.txt', 'r')
        filenameArray = listOfFiles.readlines()
        listOfAsts =  open('python-asts.json', 'r')
        astArray = listOfAsts.readlines() 
        filename+="\n"
        idxOfFile = filenameArray.index(filename)
        astOfProgram = astArray[idxOfFile]
    else: 
        languageType = "java"
        listOfFiles =  open('java-asts.txt', 'r')
        filenameArray = listOfFiles.readlines()
        listOfAsts =  open('java-asts.json', 'r')
        astArray = listOfAsts.readlines() 

        filename+="\n"
        idxOfFile = filenameArray.index(filename)
        astOfProgram = astArray[idxOfFile]

    encodedMatrix = oneHotEncoder(astOfProgram,languageType)
    adjacencyMatrix,num_nodes = adjacencyMatrixCreator(astOfProgram)
    return adjacencyMatrix,encodedMatrix,num_nodes

def getTrainingPairs():
    trainingPairs = []
    ## read from trainPairs.txt (Java, py)
    ## read from txt (Java, py)
    listOfClones =  open('../CloneDetectionSrc/ClonePairs.txt', 'r')
    listOfNonClones =  open('../CloneDetectionSrc/nonClonePairs.txt', 'r')

    trainingPairs = listOfClones.readlines()
    nonCloneTrainingPairs = listOfNonClones.readlines()


    return trainingPairs, nonCloneTrainingPairs


class PairData(Data):
    def __inc__(self, key, value):
        if key == 'edge_index1':
            return self.x1.size(0)
        if key == 'edge_index2':
            return self.x2.size(0)
        else:
            return super(PairData, self).__inc__(key, value)
    def __cat_dim__(self, key, value):
        if 'index' in key or 'face' in key:
            return 1
        else:
            return 0

    # will return dataset pairs => 
    # ------------------------------------
    # data_point -> ASTAdjacencyMatrices + encodedMatrices  + Label (pair or not)
    # ------------------------------------

## Define n
n=51917+51917
# n=4405
# n=60
class TrainLoadData(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(TrainLoadData, self).__init__(root, transform, pre_transform)
        # self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['../CloneDetectionSrc/NonClonePairs.txt']

    @property
    def processed_file_names(self):
        return ['cloneDetectionData/data_{}.pt'.format(i) for i in range(n)]

    def process(self):
        #get all the pairs - self.raw_paths
        # check pair or not and then save the pair in the data
        clonePairs,nonClonePairs = getTrainingPairs()

        if(len(clonePairs)>int(n/2)):
            clonePairs=clonePairs[:(int(n/2))]
        if(len(nonClonePairs)>int(n/2)):
            nonClonePairs=nonClonePairs[:(int(n/2))]

        i = 0
        for pairs in clonePairs:
            print(i)
            matrix1, encode1, num_nodes1 = graphToMatrices(pairs.split(",")[0])
            matrix2, encode2, num_nodes2 = graphToMatrices(pairs.split(",")[1][:-1])
            listForLabel=[1]
            labelTensor=torch.Tensor(listForLabel)
            # data = Data(x1=torch.Tensor(encode1), x2=torch.Tensor(encode2), edge_index1=torch.Tensor(matrix1), edge_index2=torch.Tensor(matrix2),num_nodes1=num_nodes1,num_nodes2=num_nodes2 y=1)
            data = PairData(x1=torch.Tensor(encode1), x2=torch.Tensor(encode2), edge_index1=torch.Tensor(matrix1), edge_index2=torch.Tensor(matrix2), y=labelTensor)
            # data1 = Data(x=torch.Tensor(encode1), edge_index=torch.LongTensor(matrix1), num_nodes=num_nodes1)
            # data2 = Data(x=torch.Tensor(encode2), edge_index=torch.LongTensor(matrix2), num_nodes=num_nodes2)
            # data = Data(data1=data1, data2=data2, y=1)
            torch.save(data, osp.join(self.processed_dir, 'cloneDetectionData/data_{}.pt'.format(i)))
            i += 1

        for pairs in nonClonePairs:
            print(i)
            matrix1, encode1, num_nodes1 = graphToMatrices(pairs.split(",")[0])
            matrix2, encode2, num_nodes2 = graphToMatrices(pairs.split(",")[1][:-1])
            listForLabel=[0]
            labelTensor=torch.Tensor(listForLabel)
            # data = Data(x1=torch.Tensor(encode1), x2=torch.Tensor(encode2), edge_index1=torch.Tensor(matrix1), edge_index2=torch.Tensor(matrix2),num_nodes1=num_nodes1,num_nodes2=num_nodes2 y=0)
            data = PairData(x1=torch.Tensor(encode1), x2=torch.Tensor(encode2), edge_index1=torch.Tensor(matrix1), edge_index2=torch.Tensor(matrix2), y=labelTensor)
            # data1 = Data(x=torch.Tensor(encode1), edge_index=torch.LongTensor(matrix1), num_nodes=num_nodes1)
            # data2 = Data(x=torch.Tensor(encode2), edge_index=torch.LongTensor(matrix2), num_nodes=num_nodes2)
            # data = Data(data1=data1, data2=data2, y=0)
            torch.save(data, osp.join(self.processed_dir, 'cloneDetectionData/data_{}.pt'.format(i)))
            i += 1
            
    def len(self):
        return len(self.processed_file_names)
    
    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'cloneDetectionData/data_{}.pt'.format(idx))) 
        # print(data)
        # print(data.edge_index2.shape)
        return data


class MyTransform(object):
    def __call__(self, data):
        # Specify target - in our case its 0 only
        data.y = data.y[:, target]
        return data

class Complete(object):
    def __call__(self, data):
        device = data.edge_index.device

        row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        col = torch.arange(data.num_nodes, dtype=torch.long, device=device)

        row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        col = col.repeat(data.num_nodes)
        edge_index = torch.stack([row, col], dim=0)

        edge_attr = None
        if data.edge_attr is not None:
            idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
            size = list(data.edge_attr.size())
            size[0] = data.num_nodes * data.num_nodes
            edge_attr = data.edge_attr.new_zeros(size)
            edge_attr[idx] = data.edge_attr

        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        data.edge_attr = edge_attr
        data.edge_index = edge_index

        return data

def train(epoch, use_unsup_loss):
    model.train()
    loss_all = 0
    sup_loss_all = 0
    unsup_loss_all = 0
    unsup_sup_loss_all = 0

    if use_unsup_loss:
        for data, udata in zip(train_loader, unsup_train_loader):
            data = data.to(device)
            udata = udata.to(device)
            optimizer.zero_grad()
            criterion = BCEWithLogitsLoss()
            pred=model(data)
            sup_loss = criterion(pred, data.y)
            unsup_loss1 = model.unsup_loss1(udata,udata.x1_batch) # unsup loss for java encoder
            unsup_loss2 = model.unsup_loss2(udata,udata.x2_batch) # unsup loss for python encoder

            if separate_encoder:
                unsup_sup_loss1 = model.unsup_sup_loss1(udata,udata.x1_batch)
                unsup_sup_loss2 = model.unsup_sup_loss2(udata,udata.x2_batch)
                loss = sup_loss + (unsup_loss1 + unsup_loss2) + (unsup_sup_loss1 + unsup_sup_loss2)* lamda
            else:
                loss = sup_loss + (unsup_loss1 + unsup_loss2 )* lamda

            loss.backward()

            sup_loss_all += sup_loss.item()*batch_size
            unsup_loss_all += (unsup_loss1.item()+unsup_loss2.item())*batch_size
            if separate_encoder:
                unsup_sup_loss_all += (unsup_sup_loss1.item()+unsup_sup_loss2.item())
            
            loss_all += loss.item() * batch_size

            optimizer.step()
        if separate_encoder:
            print(sup_loss_all, unsup_loss_all, unsup_sup_loss_all)
            return loss_all / len(train_loader.dataset),sup_loss_all, unsup_loss_all, unsup_sup_loss_all

        else:
            print(sup_loss_all/ len(train_loader.dataset), unsup_loss_all/ len(train_loader.dataset))
            return loss_all / len(train_loader.dataset),sup_loss_all/ len(train_loader.dataset), unsup_loss_all/ len(train_loader.dataset)

    else:
        cnt=0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            criterion = BCEWithLogitsLoss()
            pred=model(data)
            sup_loss = criterion(pred, data.y)
            loss = sup_loss
            loss.backward()
            loss_all += loss.item() * batch_size
            optimizer.step()
            cnt+=1
            print(cnt)

        return loss_all / len(train_loader.dataset)

def test(loader):
    model.eval()
    error = 0
    precision =0 
    recall = 0
    accuracy = 0
    predictions=[]
    grndTruth=[]
    ####################### fix the metrics 

    for data in loader:
        data = data.to(device)
        # data.x=data.x.float()
        # error += (model(data) * std - data.y * std).abs().sum().item()  # MAE
        y_pred=torch.round(torch.sigmoid(model(data)))
        predictions.append(y_pred.cpu().detach().numpy().tolist())
        grndTruth.append(data.y.cpu().detach().numpy().tolist())
        error += (y_pred - data.y).abs().sum().item()  # MAE
    
    predictions = list(itertools.chain.from_iterable(predictions))
    grndTruth = list(itertools.chain.from_iterable(grndTruth))
    accuracy += accuracy_score(grndTruth,predictions)
    precision += precision_score(grndTruth,predictions)
    recall += recall_score(grndTruth,predictions)
    
    return error / len(loader.dataset),accuracy,precision,recall

def seed_everything(seed=1234):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    seed_everything()
    from baseLineModel import Net

    # ============
    # Hyperparameters
    # ============
    target = 0
    dim = 64
    epochs = 25
    batch_size = 200
    lamda = 0.001
    use_unsup_loss = False
    separate_encoder = False

    tb = SummaryWriter()

    ## If transformation is required : 
    # transform = T.Compose([MyTransform(), Complete()])
    # dataset = JavaClassificationDataset(root="./",transform=transform)
    dataset = TrainLoadData(root="./")
    dataset=dataset.shuffle()
    print('num_features : {}\n'.format(dataset.num_features))
    dataset_num_features = max(dataset.num_features, 1)

    # if dataset.data.x is None:
    #     max_degree = 0
    #     degs = []
    #     for data in dataset:
    #         degs += [degree(data.edge_index[0], dtype=torch.long)]
    #         max_degree = max(max_degree, degs[-1].max().item())

    #     if max_degree < 1000:
    #         dataset.transform = T.OneHotDegree(max_degree)
    #     else:
    #         deg = torch.cat(degs, dim=0).to(torch.float)
    #         mean, std = deg.mean().item(), deg.std().item()
    #         dataset.transform = NormalizedDegree(mean, std)


    # Normalize targets to mean = 0 and std = 1.
    # mean = dataset.data.y[:, target].mean().item()
    # std = dataset.data.y[:, target].std().item()
    # dataset.data.y[:, target] = (dataset.data.y[:, target] - mean) / std


    ####### Split datasets.
    # trainSize=int(0.6*len(dataset))
    # valSize=int(0.2*len(dataset))
    # testSize=len(dataset)-trainSize-valSize
    # train_dataset, test_dataset , val_dataset = torch.utils.data.random_split(dataset, [trainSize,valSize,testSize])

    testLimit=int(0.2*n)
    valLimit=int(0.4*n) 
    test_dataset = dataset[:testLimit]
    val_dataset = dataset[testLimit:valLimit]
    train_dataset = dataset[valLimit:]

    test_loader = DataLoader(test_dataset, follow_batch=['x1', 'x2'],batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset,follow_batch=['x1', 'x2'], batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(train_dataset,follow_batch=['x1', 'x2'], batch_size=batch_size, shuffle=True)


    if use_unsup_loss:
        unsup_train_dataset = dataset[valLimit:]
        unsup_train_loader = DataLoader(unsup_train_dataset,follow_batch=['x1', 'x2'], batch_size=batch_size, shuffle=True)
        print(len(train_dataset), len(val_dataset), len(test_dataset), len(unsup_train_dataset))
    else:
        print(len(train_dataset), len(val_dataset), len(test_dataset))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_num_features=142 
    model = Net(dataset_num_features, dim, use_unsup_loss, separate_encoder).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5, min_lr=0.000001)
    # val_error = test(val_loader)
    # test_error = test(test_loader)
    # print('Epoch: {:03d}, Validation MAE: {:.7f}, Test MAE: {:.7f},'.format(0, val_error, test_error))
    # state=torch.load("javaModels/Classifier-Infograph*_271.pth")
    # model.load_state_dict(state['state_dict'])
    # print(state['state_dict'])

    # test_loader2 = DataLoader(test_dataset, follow_batch=['x1', 'x2'],batch_size=1000, shuffle=True)
    # for data in (test_loader2):
    #     print(data.y)

    best_val_error = None
    for epoch in range(1, epochs):
        start_time = time.time()
        lr = scheduler.optimizer.param_groups[0]['lr']
        print("Training epoch %d :" % epoch)
        if separate_encoder:
            loss,sup_loss, unsup_loss, unsup_sup_loss = train(epoch, use_unsup_loss)
        else:
            if use_unsup_loss:
                loss,sup_loss, unsup_loss = train(epoch, use_unsup_loss)
            else:
                sup_loss = train(epoch, use_unsup_loss)
            
        print("Testing epoch %d :" % epoch)

        val_error,val_accuracy,val_prec,val_recall = test(val_loader)
        scheduler.step(val_error)

        if best_val_error is None or val_error <= best_val_error:
            test_error,test_accuracy,test_prec,test_recall = test(test_loader)
            best_val_error = val_error

        tb.add_scalar('Sup_Loss', sup_loss, epoch)
        if use_unsup_loss:
            tb.add_scalar('UnSup Loss', unsup_loss, epoch)
        if separate_encoder:
            tb.add_scalar('UnSup-Sup- Loss', unsup_sup_loss, epoch)
        tb.add_scalar('val_error', val_error, epoch)
        tb.add_scalar('val_accuracy', val_accuracy, epoch)
        tb.add_scalar('val_prec', val_prec, epoch)
        tb.add_scalar('val_recall', val_recall, epoch)
        tb.add_scalar('test_error', test_error, epoch)
        tb.add_scalar('test_accuracy', test_accuracy, epoch)
        tb.add_scalar('test_prec', test_prec, epoch)
        tb.add_scalar('test_recall', test_recall, epoch)

        end_time = time.time()
        epoch_duration = (end_time - start_time)/3600
        # change this
        with open('cloneDetection-EpochResults-baseLine-SimpleGCn.txt', 'a+') as f:
            if separate_encoder:
                f.write('Epoch: {:03d}, LR: {:7f}, T.Loss: {:.7f}, Sup-Loss: {:.7f},unSup-Loss: {:.7f},unSup-sup-Loss: {:.7f}, Validation MAE: {:.7f},Validation Acc: {:.7f},Validation Prec: {:.7f},Validation Rec: {:.7f},Test MAE: {:.7f},Test Acc: {:.7f},Test Prec: {:.7f},Test Rec: {:.7f}, Time : {:.7f}'.format(epoch, lr, loss,sup_loss, unsup_loss, unsup_sup_loss, val_error,val_accuracy,val_prec,val_recall,test_error,test_accuracy,test_prec,test_recall,epoch_duration))
            else:
                if use_unsup_loss:
                    f.write('Epoch: {:03d}, LR: {:7f}, T.Loss: {:.7f}, Sup-Loss: {:.7f},unSup-Loss: {:.7f}, Validation MAE: {:.7f},Validation Acc: {:.7f},Validation Prec: {:.7f},Validation Rec: {:.7f},Test MAE: {:.7f},Test Acc: {:.7f},Test Prec: {:.7f},Test Rec: {:.7f}, Time : {:.7f}'.format(epoch, lr, loss,sup_loss, unsup_loss, val_error,val_accuracy,val_prec,val_recall,test_error,test_accuracy,test_prec,test_recall,epoch_duration))
                else:
                    f.write('Epoch: {:03d}, LR: {:7f}, Sup-Loss: {:.7f}, Validation MAE: {:.7f},Validation Acc: {:.7f},Validation Prec: {:.7f},Validation Rec: {:.7f},Test MAE: {:.7f},Test Acc: {:.7f},Test Prec: {:.7f},Test Rec: {:.7f}, Time : {:.7f}'.format(epoch, lr,sup_loss, val_error,val_accuracy,val_prec,val_recall,test_error,test_accuracy,test_prec,test_recall,epoch_duration))

            f.write('\n')

        # change this
        torch.save({'state_dict': model.state_dict(),'optimizer' : optimizer.state_dict()},"cloneDetectionModels/baseline_SimpleGCN_bigDB_new_"+str(epoch)+".pth")

        # change this
    # with open('cloneDetection-EpochResults-baseLine-SimpleGCn.log', 'a+') as f:
    #     f.write('{},{},{},{},{},{},{},{}\n'.format(target,1000,use_unsup_loss,separate_encoder,0.001,0,val_error,test_error))
    #     f.write('\n')
    #     f.write("Total time taken for evaluation is: ",(end_time - start_time)/3600,"hrs.")
    
    tb.close()