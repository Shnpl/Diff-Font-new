import sys
sys.path.append('..')
import csv
from torchvision.models import vit_b_32,ViT_B_32_Weights
import numpy as np
from types import MethodType

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pytorch_lightning.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT, TRAIN_DATALOADERS
import pytorch_lightning as pl
from pytorch_lightning.core import LightningModule

from improved_diffusion.image_datasets import ImageDataset

class FontClassifier_LitModel(LightningModule):
    def __init__(self,model_type='resnet50',data_type='800'):
        super().__init__()
        self.model_type = model_type
        self.data_type = data_type
        if model_type == 'resnet18':
            self.model = FontClassifier_resnet18()
            self.hidden_dim = 512
        elif model_type == 'resnet34':
            self.model = FontClassifier_resnet34()
            self.hidden_dim = 512
        elif model_type == 'resnet50':
            self.model = FontClassifier_resnet50()
            self.hidden_dim = 2048
        elif model_type == 'vit_b_32':
            self.model = FontClassifier_ViT_B_32()
            self.hidden_dim = 768
        self.loss = nn.CrossEntropyLoss()
    def forward(self,x):
        return self.model(x)

    def configure_optimizers(self):
        opt = optim.Adam(self.model.parameters())
        return opt
    def on_train_epoch_start(self) -> None:
        self.train_count_idx = 0
        self.train_loss = torch.zeros((self.train_dataset_length,1),dtype=torch.float32)
        # self.train_result_labels = torch.zeros((self.val_dataset_length,1),dtype=torch.int64)
    def training_step(self, batch, batch_idx):
        x = batch[1]['style_image'].float()
        y = batch[1]['style_name']
        y_gt = torch.zeros((x.shape[0],499),dtype=torch.float32)
        bs = len(y_gt)
        for i,y_i in enumerate(y):
            y_gt[i,int(y_i)] = 1
        
        y_pred = self.forward(x)
        y_gt = y_gt.to(self.device)
        loss = self.loss(y_pred,y_gt)
        
        self.train_loss[self.train_count_idx:self.train_count_idx+bs] = loss.detach().cpu()
        self.train_count_idx += bs
        return loss
    def on_train_epoch_end(self) -> None:
        self.log('train_loss',torch.mean(self.train_loss))      
    def on_validation_epoch_start(self) -> None:
        self.val_count_idx = 0
        
        self.val_result = torch.zeros((self.val_dataset_length,self.hidden_dim),dtype=torch.float32)
        self.val_result_labels = ['']*self.val_dataset_length
    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT | None:        
        x = batch[1]['style_image'].float()
        y_gt = batch[1]['style_name']

        bs = len(y_gt)

        self.model.use_fc = False
        y_pred_hidden = self.forward(x)
        self.model.use_fc = True
        
        # for i in range(len(y_gt)):
        self.val_result_labels[self.val_count_idx:self.val_count_idx+bs] = y_gt
        self.val_result[self.val_count_idx:self.val_count_idx+bs,:] = y_pred_hidden.detach().cpu()
        self.val_count_idx += bs

        
    def on_validation_epoch_end(self) -> None:
        results:torch.Tensor = self.val_result
        labels = self.val_result_labels
        unique_labels = []
        for label in labels:
            if label not in unique_labels:
                unique_labels.append(label)
        unique_label_idces_list = []
        for label in unique_labels:
            unique_label_idces_list.append([i for i in range(len(labels)) if labels[i] == label])
        dim = results.shape[1]
        N_types = len(unique_labels)
        s_w = torch.zeros((dim,dim),dtype=torch.float32)
        m_w = torch.zeros((N_types,dim),dtype=torch.float32)
        for i,unique_label_idces in enumerate(unique_label_idces_list):
            s_w_results :torch.Tensor = results[unique_label_idces]
            cov = torch.cov(s_w_results.T)
            s_w += cov
            m_w[i] = torch.mean(s_w_results,dim=0)

        s_b = torch.cov(m_w.T)*(N_types-1)
        val_measure = torch.trace(s_w/N_types)/torch.trace(s_b)
        self.log('val_measure', val_measure,sync_dist=True)
        
            
    def test_step(self, batch, batch_idx):
        if not hasattr(self,'result'):
            self.result = np.zeros((self.test_dataset_length,self.hidden_dim))
            self.count = 0
        if not hasattr(self,'result_label'):
            self.result_label = []
            self.count = 0
        x = batch[1]['style_image'].float().to(self.device)
        y_gt = batch[1]['style_name']

        
        y_pred_hidden = self.forward(x)
        # with open('out.csv','a') as csvfile:
        #     writer = csv.writer(csvfile)
        bs = len(y_gt)
        # for i in range(len(y_gt)):
        self.result_label.extend(y_gt)
        self.result[self.count:self.count+bs,:] = y_pred_hidden.detach().cpu()
        self.count += bs
        # log 6 example images
        # or generated text... or whatever

        # calculate acc
    #   y_pred = torch.argmax(y_pred, dim=1)

    #     test_acc = torch.sum(y_pred == y_gt).item() / (len(y_gt) * 1.0)

    #     # log the outputs!
    #     self.log_dict({'test_loss': loss, 'test_acc': test_acc})
    #     self.log_dict({'test_loss': loss, 'test_acc': test_acc})

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        if self.data_type == '800':
            train_dataset = ImageDataset(data_dir='datasets/CFG/font500_800',stroke_path=None)
        elif self.data_type == '6763':
            train_dataset = ImageDataset(data_dir='datasets/CFG/font500_6763',stroke_path=None)
        self.train_dataset_length = len(train_dataset)
        return DataLoader(train_dataset,shuffle=True,num_workers=20,batch_size=16)
    def val_dataloader(self) -> EVAL_DATALOADERS:
        if self.data_type == '800':
            val_dataset = ImageDataset(data_dir='datasets/CFG/font_extra_800_val',stroke_path=None)
        elif self.data_type == '6763':
            val_dataset = ImageDataset(data_dir='datasets/CFG/font_extra_800_val',stroke_path=None)
        self.val_dataset_length = len(val_dataset)
        return DataLoader(val_dataset,shuffle=False,num_workers=20,batch_size=16)
    def test_dataloader(self) -> EVAL_DATALOADERS:
        if self.data_type == '800':
            test_dataset = ImageDataset(data_dir='datasets/CFG/font_extra_800_test',stroke_path=None)
        elif self.data_type == '6763':
            test_dataset = ImageDataset(data_dir='datasets/CFG/font_extra_800_test',stroke_path=None)
        self.test_dataset_length = len(test_dataset)
        return DataLoader(test_dataset,shuffle=False,num_workers=20,batch_size=16)

class FontClassifier_resnet18(nn.Module):
    def __init__(self,use_fc = True):
        super().__init__()
        # 3*128*128->64*120*120->64*112*112
        self.conv1  = nn.Sequential(
            nn.Conv2d(3,64,9),
            nn.Conv2d(64,64,9),
        )
        self.pool1 = nn.MaxPool2d(3,2)#64*112*112->64*56*56
        #64*56*56->64*56*56
        self.resblock2 = nn.Sequential(
                PreActResNetBlock(64,nn.ReLU),
                PreActResNetBlock(64,nn.ReLU)
        )
        #64*56*56->128*28*28
        self.resblock3 = nn.Sequential(
                PreActResNetBlock(64,nn.ReLU,True,128),
                PreActResNetBlock(128,nn.ReLU)
        )
        #128*28*28->256*14*14
        self.resblock4 = nn.Sequential(                
                PreActResNetBlock(128,nn.ReLU,True,256),
                PreActResNetBlock(256,nn.ReLU)
        )
        #256*14*14->512*7*7
        self.resblock5 = nn.Sequential(
                PreActResNetBlock(256,nn.ReLU,True,512),
                PreActResNetBlock(512,nn.ReLU)
        )
        #512*7*7->512*1*1
        self.pool_final = nn.AdaptiveAvgPool2d((1,1))
        self.fc_final = nn.Linear(512,499)
        self.use_fc = use_fc
    def forward(self,x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.resblock5(x)
        x = torch.squeeze(torch.squeeze(self.pool_final(x),-1),-1)
        if self.use_fc:
            x = self.fc_final(x)
        return x

class FontClassifier_resnet34(nn.Module):
    def __init__(self,use_fc = True):
        super().__init__()
        # 3*128*128->64*120*120->64*112*112
        self.conv1  = nn.Sequential(
            nn.Conv2d(3,64,9),
            nn.Conv2d(64,64,9),
        )
        # 64*112*112->64*56*56
        self.pool1 = nn.MaxPool2d(3,2)
        # 64*56*56->64*56*56
        self.resblock2 = nn.Sequential(
                PreActResNetBlock(64,nn.ReLU),
                PreActResNetBlock(64,nn.ReLU),
                PreActResNetBlock(64,nn.ReLU)
        )
        # 64*56*56->128*28*28
        self.resblock3 = nn.Sequential(
                PreActResNetBlock(64,nn.ReLU,True,128),
                PreActResNetBlock(128,nn.ReLU),
                PreActResNetBlock(128,nn.ReLU),
                PreActResNetBlock(128,nn.ReLU)
        )
        # 128*28*28->256*14*14
        self.resblock4 = nn.Sequential(
                PreActResNetBlock(128,nn.ReLU,True,256),
                PreActResNetBlock(256,nn.ReLU),
                PreActResNetBlock(256,nn.ReLU),
                PreActResNetBlock(256,nn.ReLU),
                PreActResNetBlock(256,nn.ReLU),
                PreActResNetBlock(256,nn.ReLU),
        )
        # 256*14*14->512*7*7
        self.resblock5 = nn.Sequential(
                PreActResNetBlock(256,nn.ReLU,True,512),
                PreActResNetBlock(512,nn.ReLU),
                PreActResNetBlock(512,nn.ReLU)
        )
        # 512*7*7->512*1*1
        self.pool_final = nn.AdaptiveAvgPool2d((1,1))
        self.fc_final = nn.Linear(512,499)
        self.use_fc = use_fc
    def forward(self,x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.resblock5(x)
        x = torch.squeeze(torch.squeeze(self.pool_final(x),-1),-1)
        if self.use_fc:
            x = self.fc_final(x)
        return x

class FontClassifier_resnet50(nn.Module):
    def __init__(self,use_fc = True):
        super().__init__()
        # 3*128*128->64*120*120->64*112*112
        self.conv1  = nn.Sequential(
            nn.Conv2d(3,64,9),
            nn.Conv2d(64,64,9),
        )
        # 64*112*112->64*56*56
        self.pool1 = nn.MaxPool2d(3,2)
        # 64*56*56->64*56*56
        self.resblock2 = nn.Sequential(
                ResNetBottleneckBlock(64,64,nn.ReLU,False,256),
                ResNetBottleneckBlock(256,64,nn.ReLU,False,256),
                ResNetBottleneckBlock(256,64,nn.ReLU,False,256)
        )
        # 64*56*56->128*28*28
        self.resblock3 = nn.Sequential(
                ResNetBottleneckBlock(256,128,nn.ReLU,True,512),
                ResNetBottleneckBlock(512,128,nn.ReLU,False,512),
                ResNetBottleneckBlock(512,128,nn.ReLU,False,512),
                ResNetBottleneckBlock(512,128,nn.ReLU,False,512)
        )
        # 128*28*28->256*14*14
        self.resblock4 = nn.Sequential(
                ResNetBottleneckBlock(512,256,nn.ReLU,True,1024),
                ResNetBottleneckBlock(1024,256,nn.ReLU,False,1024),
                ResNetBottleneckBlock(1024,256,nn.ReLU,False,1024),
                ResNetBottleneckBlock(1024,256,nn.ReLU,False,1024),
                ResNetBottleneckBlock(1024,256,nn.ReLU,False,1024),
                ResNetBottleneckBlock(1024,256,nn.ReLU,False,1024)
            )
        # 256*14*14->512*7*7
        self.resblock5 = nn.Sequential(
                ResNetBottleneckBlock(1024,512,nn.ReLU,True,2048),
                ResNetBottleneckBlock(2048,512,nn.ReLU,False,2048),
                ResNetBottleneckBlock(2048,512,nn.ReLU,False,2048)
            )
        # 512*7*7->512*1*1
        self.pool_final = nn.AdaptiveAvgPool2d((1,1))
        self.fc_final = nn.Linear(2048,499)
        self.use_fc = use_fc
    def forward(self,x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.resblock5(x)
        x = torch.squeeze(torch.squeeze(self.pool_final(x),-1),-1)
        if self.use_fc:
            x = self.fc_final(x)
        return x
class PreActResNetBlock(nn.Module):
    def __init__(self, c_in, act_fn, subsample=False, c_out=-1):
        """
        Inputs:
            c_in - Number of input features
            act_fn - Activation class constructor (e.g. nn.ReLU)
            subsample - If True, we want to apply a stride inside the block and reduce the output shape by 2 in height and width
            c_out - Number of output features. Note that this is only relevant if subsample is True, as otherwise, c_out = c_in
        """
        super().__init__()
        if not subsample:
            c_out = c_in

        # Network representing F
        self.net = nn.Sequential(
            nn.BatchNorm2d(c_in),
            act_fn(),
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, stride=1 if not subsample else 2, bias=False),
            nn.BatchNorm2d(c_out),
            act_fn(),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
        )

        # 1x1 convolution needs to apply non-linearity as well as not done on skip connection
        self.downsample = (
            nn.Sequential(nn.BatchNorm2d(c_in), act_fn(), nn.Conv2d(c_in, c_out, kernel_size=1, stride=2, bias=False))
            if subsample
            else None
        )

    def forward(self, x):
        z = self.net(x)
        if self.downsample is not None:
            x = self.downsample(x)
        out = z + x
        return out

class ResNetBottleneckBlock(nn.Module):
    def __init__(self, c_in,c_internal, act_fn, subsample=False, c_out=-1):
        """
        Inputs:
            c_in - Number of input features
            act_fn - Activation class constructor (e.g. nn.ReLU)
            subsample - If True, we want to apply a stride inside the block and reduce the output shape by 2 in height and width
            c_out - Number of output features. Note that this is only relevant if subsample is True, as otherwise, c_out = c_in
        """
        super().__init__()

        # Network representing F
        self.net = nn.Sequential(
            nn.BatchNorm2d(c_in),
            act_fn(),
            nn.Conv2d(c_in, c_internal, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(c_internal),
            act_fn(),
            nn.Conv2d(c_internal, c_internal, kernel_size=3, padding=1, stride=1 if not subsample else 2, bias=False),
            nn.BatchNorm2d(c_internal),
            act_fn(),
            nn.Conv2d(c_internal, c_out, kernel_size=1,stride=1, bias=False),
        )

        # 1x1 convolution needs to apply non-linearity as well as not done on skip connection
        self.downsample = (
            nn.Sequential(nn.BatchNorm2d(c_in), act_fn(), nn.Conv2d(c_in, c_out, kernel_size=1, stride=2, bias=False))
            if subsample
            else nn.Sequential(nn.BatchNorm2d(c_in), act_fn(), nn.Conv2d(c_in, c_out, kernel_size=1, stride=1, bias=False))
        )

    def forward(self, x):
        z = self.net(x)
        x = self.downsample(x)
        out = z + x
        return out

class FontClassifier_ViT_B_32(nn.Module):
    def __init__(self,use_fc = True) -> None:
        super().__init__()
        weights = ViT_B_32_Weights.DEFAULT
        self.preprocess = weights.transforms()
        self.vit = vit_b_32(weights=weights, progress=True)
        delattr(self.vit, "heads")
        self.vit.forward = MethodType(customized_vit_forward, self.vit)
        self.fc = nn.Linear(768,499)
        self.use_fc = use_fc
    def forward(self,x):
        x = self.preprocess(x)
        x = self.vit(x)
        if self.use_fc:
            x = self.fc(x)
        return x
def customized_vit_forward(self, x: torch.Tensor):
    # Reshape and permute the input tensor
    x = self._process_input(x)
    n = x.shape[0]

    # Expand the class token to the full batch
    batch_class_token = self.class_token.expand(n, -1, -1)
    x = torch.cat([batch_class_token, x], dim=1)

    x = self.encoder(x)

    # Classifier "token" as used by standard language architectures
    x = x[:, 0]

    return x

if __name__ == "__main__":
    cla =  FontClassifier_ViT_B_32()
    input = torch.randn((1,3,128,128))
    output = cla(input)
    print(output.shape)