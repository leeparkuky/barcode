from memory_profiler import profile
import sys
import os
import pandas as pd
import numpy as np
# sys.path.append('/root/barcode/')
sys.path.append('../')
from BarcodeScanner import tree_and_clustering, base_barcode
from itertools import product, combinations
from sklearn.linear_model import LinearRegression
import timeit
from datasets import Dataset
from itertools import product
from torch.utils.data import DataLoader
import torch
from torch import nn
import torch.optim as optim




class Lasso_Barcode(nn.Module):
    def __init__(self, num_variable):
        super().__init__()
        barcode_size = 2**num_variable
        embedding_weights = torch.from_numpy(L(num_variable).astype(float))
        self.embedding = nn.Embedding(barcode_size, barcode_size)
        self.embedding.weight = nn.Parameter(embedding_weights.to(torch.float32), requires_grad = False)
        self.linear = nn.Linear(barcode_size, 1, bias = False, dtype = torch.float32)

    def l1_reg(self):
        return torch.abs(self.linear.weight).sum()

    def forward(self, x):
        x = self.embedding(x)
        x = self.linear(x)
        return x, self.l1_reg()




def L(p):
    all_sets = list(set(product([0,1], repeat = p))); all_sets.sort()
    return np.array([base_barcode.barcode_to_beta(x) for x in all_sets]).astype(np.int8)


def gen_X(num_var: int, sample_size : int):
    data_dictionary = {}
    for i in range(num_var):
        var_name = "x" + f"{i + 1}"
        data_dictionary[var_name] = list(np.random.binomial(1, .5, sample_size))
    return pd.DataFrame(data_dictionary)

def gen_full_X(num_var: int, sample_size :int):
    raw_X = gen_X(num_var = num_var, sample_size = sample_size)
    colnames = raw_X.columns
    for k in range(2, len(colnames)+ 1):
        interaction_generator = combinations(colnames, k)
        for interaction_tuple in interaction_generator:
            new_colname = "*".join(interaction_tuple)
            raw_X[new_colname] = raw_X[list(interaction_tuple)].apply(np.prod, axis = 1)
    return raw_X

def gen_barcode_dataloader(num_var:int, sample_size:int):
    raw_X = gen_X(num_var = num_var, sample_size = sample_size)
    colnames = [
        f"x{i+1}" for i in range(num_var)
    ]
    dataset = Dataset.from_pandas(raw_X)
    def gen_z(examples):
        example_list = [examples[x] for x in colnames]
        df = pd.DataFrame(zip(*example_list), columns= colnames)
        barcodes = base_barcode.gen_barcode(df).reshape(-1).tolist()
        y = df.apply(lambda seq: 1 + seq.x1 + seq.x2 + seq.x1*seq.x3 + np.random.normal(), axis = 1).tolist()
        return {"z": barcodes, "y":y}
    
    dataset = dataset.map(gen_z, batched = True, remove_columns=colnames, num_proc = 10)
    return dataset



def train(train_dataloader, lasso, optimizer, alpha = 0.3, criterion = nn.MSELoss(), device = None):
    alpha = 0.3
    lasso.train()
    for batch in train_dataloader:
        input_tensor = batch['z']
        output_tensor = batch['y'].to(torch.float32)
        output_tensor = torch.reshape(output_tensor, (-1, 1))

        if device:
            input_tensor = input_tensor.to(device)
            output_tensor = output_tensor.to(device)

        optimizer.zero_grad()
        outputs, l1_reg = lasso(input_tensor)
        # loss = criterion(outputs, output_feature)

        loss = criterion(outputs, output_tensor) + alpha * l1_reg  # Total loss with L1 regularization
        loss.backward()
        optimizer.step()


def evaluate(test_dataloader, lasso, alpha = 0.3, criterion = nn.MSELoss(), device = None):
    lasso.eval()
    losses = []
    for batch in test_dataloader:
        input_tensor = batch['z']
        output_tensor = batch['y'].to(torch.float32)
        output_tensor = torch.reshape(output_tensor, (-1, 1))
        if device:
            input_tensor = input_tensor.to(device)
            output_tensor = output_tensor.to(device)
        outputs, l1_reg = lasso(input_tensor)
        loss = criterion(outputs, output_tensor) + alpha * l1_reg  # Total loss with L1 regularization
        if device:
            loss = loss.to(torch.device('cpu'))
        losses.append(loss.item())
    current_val_loss = np.mean(losses)
    return current_val_loss


from tqdm import tqdm
@profile
def pipeline(p, n, input_dataset):
    device = torch.device('cuda')
    input_dataset = input_dataset.train_test_split(test_size = .2)

    train_dataloader = DataLoader(input_dataset['train'], batch_size=2**13, shuffle=True)
    test_dataloader = DataLoader(input_dataset['test'], batch_size=2**14, shuffle=True)


    lasso = Lasso_Barcode(p).to(device)
    optimizer = optim.Adam(lasso.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    num_epochs = 10
    val_loss = np.inf
    for epoch in tqdm(range(num_epochs)):
        # Forward pass
        alpha = 0.3
        lasso.train()
        for batch in train_dataloader:
            input_tensor = batch['z']
            output_tensor = batch['y'].to(torch.float32)
            output_tensor = torch.reshape(output_tensor, (-1, 1))

            if device:
                input_tensor = input_tensor.to(device)
                output_tensor = output_tensor.to(device)

            optimizer.zero_grad()
            outputs, l1_reg = lasso(input_tensor)
            # loss = criterion(outputs, output_feature)

            loss = criterion(outputs, output_tensor) + alpha * l1_reg  # Total loss with L1 regularization
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            current_val_loss = evaluate(test_dataloader, lasso, device = device)
            if val_loss > current_val_loss:
                val_loss = current_val_loss
            else:
                break
                
    return lasso


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", help="number of binary explanatory variables", type = int)
    parser.add_argument("-n", help= 'sample_size', type = int)
    
    args = parser.parse_args()
    input_dataset = gen_barcode_dataloader(args.p, args.n)

    
    pipeline(args.p, args.n, input_dataset)
    


