import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.checkpoint import checkpoint  
import os
#%%
class EarlyStopping():
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0,baseline=1,filename='finish_model'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            上次验证集损失值改善后等待几个epoch
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            如果是True，为每个验证集损失值改善打印一条信息
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            监测数量的最小变化，以符合改进的要求
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.baseline=baseline
        self.filename=filename
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            # if val_loss<=self.baseline:
            self.save_checkpoint(val_loss, model) 
            if val_loss<=self.baseline: #足夠好
                self.save_checkpoint(val_loss, model) #比基準好才存
                print('good enough')
                self.early_stop = True   
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''
        Saves model when validation loss decrease.
        验证损失减少时保存模型。
        '''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), 'checkpoint.pt')     # 这里会存储迄今最优模型的参数
            
        if os.path.exists(self.filename+'.pkl'):  # checking if there is a file with this name
            os.remove(self.filename+'.pkl')  # deleting the file
        torch.save(model, self.filename+'.pkl')                 # 这里会存储迄今最优的模型
        self.val_loss_min = val_loss
#%% 可分離捲機
class Sep_Conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Sep_Conv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )
        
    def seg0(self, y):
        y=self.depth_conv(y)         
        return y
    def seg1(self, y):
        y= self.point_conv(y)         
        return y


    def forward(self, input):
        y = input
        y = y + torch.zeros(1, dtype=y.dtype, device=y.device, requires_grad=True)       
        y=checkpoint(self.seg0, y)
        y=checkpoint(self.seg1, y)       
        # out = self.point_conv(out)
        return y

#%%

#(in_channels=self.input_dim + self.hidden_dim,#卷積輸入的尺寸
#                              out_channels=4 * self.hidden_dim,#因為lstmcell有四個門，隱藏層單元是rnn的四倍
#                              kernel_size=self.kernel_size,
#                              padding=self.padding,
#                              bias=self.bias)
#combinetest
#ConvLSTM(input_dim=1,
#                 hidden_dim=[64, 64, 1],
#                 kernel_size=(3, 3),
#                 num_layers=3,
#                 batch_first=True,
#                 bias=True,
#                 return_all_layers=False)
#
#(1, 65, 70, 120)#combined_conv shapetorch.Size([1, 4, 70, 120])
class InceptionBlock(nn.Module):

    def __init__(self, in_channels,out_channels,padding,bias):
        super(InceptionBlock, self).__init__()
        
        self.branch1x1 = nn.Conv2d(in_channels, 3*out_channels, kernel_size=1,bias=bias)
        
        self.branch3x3 = nn.Conv2d(in_channels, 3*out_channels, kernel_size=3, padding=padding,bias=bias)

        self.branch3x3dbl_1 = nn.Conv2d(in_channels, out_channels*5, kernel_size=1)
        self.branch3x3dbl_2 = nn.Conv2d(out_channels*5, out_channels*4, kernel_size=3, padding=padding)
        self.branch3x3dbl_3 = nn.Conv2d(out_channels*4, 3*out_channels, kernel_size=3,padding=padding)
        
        
        self.out1x1 = nn.Conv2d(out_channels*3*3, out_channels, kernel_size=1,bias=bias) #用來合併融合成 輸出成4個通道（門）

    def forward(self, x):
#        print(str(x.shape))
#        branch3x3 = F.max_pool2d(self.branch3x3(x),(2,2))
        branch3x3 = self.branch3x3(x)
        
        branch3x3 = F.relu(branch3x3)
#        print(str(branch3x3.shape))
        
        
#        branch1x1 = F.max_pool2d(self.branch1x1(x),(2,2))   
        branch1x1 = self.branch1x1(x)  
#        branch1x1 = F.max_pool2d(self.branch1x1(x),(2,2))  
#        print(str(branch1x1.shape)) 
        branch1x1 = F.relu(branch1x1)
        
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = F.relu(branch3x3dbl)
#        print(str(branch3x3dbl.shape))
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = F.relu(branch3x3dbl)
#        print(str(branch3x3dbl.shape))
#        branch3x3dbl = F.max_pool2d(self.branch3x3dbl_3(branch3x3dbl),(2,2))
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch3x3dbl = F.relu(branch3x3dbl)
#        print(str(branch3x3dbl.shape))

#        branch_pool = F.max_pool2d(x, kernel_size=3)

        outputs = [branch1x1,branch3x3, branch3x3dbl]
#        print('hi')
        
        cnn1dOut=self.out1x1(torch.cat(outputs, 1))  #
#        cnn1dOut=F.max_pool2d(self.out1x1(torch.cat(outputs, 1)),(2,2))  #pool test
#        print(str(cnn1dOut.shape))
        return cnn1dOut
    
#combinetest=torch.randn(1, 65, 70, 120)#combined_conv shapetorch.Size([1, 4, 70, 120])    
#incetest= InceptionBlock(65,4,1,True)
#outtest=    incetest(combinetest)
#outtest.shape

#aaaa=list(outtest.shape)
#%%
class InceptionBlock2(nn.Module):

    def __init__(self, in_channels,out_channels,padding,bias):
        super(InceptionBlock2, self).__init__()
        
        self.branch1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        
        self.branch3x3_1 = nn.Conv2d(in_channels, 4, kernel_size=1)
        self.branch3x3_2 = Sep_Conv(4, out_channels)

        self.branch3x3dbl_1 = nn.Conv2d(in_channels, 4, kernel_size=1)
        self.branch3x3dbl_2 = Sep_Conv(4, 8)
        self.branch3x3dbl_3 = Sep_Conv(8, out_channels)
        
        self.branch3x3db7_1 = nn.Conv2d(in_channels, 4, kernel_size=1)
        self.branch3x3db7_2 =Sep_Conv(4, 8)
        self.branch3x3db7_3 = Sep_Conv(8, 16)
        self.branch3x3db7_4 = Sep_Conv(16, out_channels)
            #7x7
        
        self.out1x1 = nn.Conv2d(out_channels*4, out_channels, kernel_size=1) #用來合併融合成 輸出成4個通道（門）





    def branch1(self, x):
        x=self.branch1x1(x)  
        x=F.layer_norm(x,x.size()[1:])
        x=F.relu(x,inplace=True)  
        return x

    def branch3(self, x):
        x = self.branch3x3_1(x)
        x=F.layer_norm(x,x.size()[1:])        
        x = F.relu(x,inplace=True)
        x = self.branch3x3_2(x)
        x=F.layer_norm(x,x.size()[1:])
        x = F.relu(x,inplace=True)
        return x


    def branch3dbl(self, x):
        x = self.branch3x3dbl_1(x)
        x=F.layer_norm(x,x.size()[1:])
        x = F.relu(x,inplace=True)
        x = self.branch3x3dbl_2(x)
        x=F.layer_norm(x,x.size()[1:])
        x = F.relu(x,inplace=True)
        x = self.branch3x3dbl_3(x)
        x=F.layer_norm(x,x.size()[1:])        
        x = F.relu(x,inplace=True)
        return x
    
    def branch3db7(self, x):
        x = self.branch3x3db7_1(x)
        x=F.layer_norm(x,x.size()[1:])        
        x = F.relu(x,inplace=True)
        x = self.branch3x3db7_2(x)
        x=F.layer_norm(x,x.size()[1:]) 
        x = F.relu(x,inplace=True)
        x = self.branch3x3db7_3(x)
        x=F.layer_norm(x,x.size()[1:]) 
        x = F.relu(x,inplace=True)
        x = self.branch3x3db7_4(x)
        x=F.layer_norm(x,x.size()[1:]) 
        x = F.relu(x,inplace=True)
        return x  


    def forward(self, x):

        # branch1x1 = self.branch1x1(x)
        # branch1x1=F.layer_norm(branch1x1,branch1x1.size()[1:])
        # branch1x1 = F.relu(branch1x1,inplace=True)        
        
        
        # branch3x3 = self.branch3x3_1(x)
        # branch3x3=F.layer_norm(branch3x3,branch3x3.size()[1:])        
        # branch3x3 = F.relu(branch3x3,inplace=True)
        # branch3x3 = self.branch3x3_2(branch3x3)
        # branch3x3=F.layer_norm(branch3x3,branch3x3.size()[1:])
        # branch3x3 = F.relu(branch3x3,inplace=True)        
        

        
        # branch3x3dbl = self.branch3x3dbl_1(x)
        # branch3x3dbl=F.layer_norm(branch3x3dbl,branch3x3dbl.size()[1:])
        # branch3x3dbl = F.relu(branch3x3dbl,inplace=True)
        # branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        # branch3x3dbl=F.layer_norm(branch3x3dbl,branch3x3dbl.size()[1:])
        # branch3x3dbl = F.relu(branch3x3dbl,inplace=True)
        # branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        # branch3x3dbl=F.layer_norm(branch3x3dbl,branch3x3dbl.size()[1:])        
        # branch3x3dbl = F.relu(branch3x3dbl,inplace=True)


        # branch3x3db7 = self.branch3x3db7_1(x)
        # branch3x3db7=F.layer_norm(branch3x3db7,branch3x3db7.size()[1:])        
        # branch3x3db7 = F.relu(branch3x3db7,inplace=True)
        # branch3x3db7 = self.branch3x3db7_2(branch3x3db7)
        # branch3x3db7=F.layer_norm(branch3x3db7,branch3x3db7.size()[1:]) 
        # branch3x3db7 = F.relu(branch3x3db7,inplace=True)
        # branch3x3db7 = self.branch3x3db7_3(branch3x3db7)
        # branch3x3db7=F.layer_norm(branch3x3db7,branch3x3db7.size()[1:]) 
        # branch3x3db7 = F.relu(branch3x3db7,inplace=True)
        # branch3x3db7 = self.branch3x3db7_4(branch3x3db7)
        # branch3x3db7=F.layer_norm(branch3x3db7,branch3x3db7.size()[1:]) 
        # branch3x3db7 = F.relu(branch3x3db7,inplace=True)

        y = x
        y = y + torch.zeros(1, dtype=y.dtype, device=y.device, requires_grad=True)       
        y1=checkpoint(self.branch1, y)
        y2=checkpoint(self.branch3, y)
        y3=checkpoint(self.branch3dbl, y)
        y4=checkpoint(self.branch3db7, y) 
        
        # outputs = [branch1x1,branch3x3, branch3x3dbl,branch3x3db7]
        outputs = [y1,y2, y3,y4]
#        print('hi')
        
        cnn1dOut=self.out1x1(torch.cat(outputs, 1))  #
#        cnn1dOut=F.max_pool2d(self.out1x1(torch.cat(outputs, 1)),(2,2))  #pool test
#        print(str(cnn1dOut.shape))
        return cnn1dOut
    
#combinetest=torch.randn(1, 65, 70, 120)#combined_conv shapetorch.Size([1, 4, 70, 120])    
#incetest= InceptionBlock(65,4,1,True)
#outtest=    incetest(combinetest)
#outtest.shape

#aaaa=list(outtest.shape)
        
#%%
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        #input_dim是每個num_layer的第一個時刻的的輸入dim，即channel
        #hidden_dim是每一個num_layer的隱藏層單元，如第一層是64，第二層是128，第三層是128
        #kernel_size是卷積核
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        #padding的目的是保持卷積之後大小不變
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

#        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,#卷積輸入的尺寸
#                              out_channels=4 * self.hidden_dim,#因為lstmcell有四個門，隱藏層單元是rnn的四倍
#                              kernel_size=self.kernel_size,
#                              padding=self.padding,
#                              bias=self.bias)
#        self.conv = InceptionBlock(in_channels=self.input_dim + self.hidden_dim,#卷積輸入的尺寸
#                              out_channels=4 * self.hidden_dim,#因為lstmcell有四個門，隱藏層單元是rnn的四倍
#                              padding=self.padding,
#                              bias=self.bias)        
        
        self.conv = InceptionBlock2(in_channels=self.input_dim + self.hidden_dim,#卷積輸入的尺寸
                              out_channels=4 * self.hidden_dim,#因為lstmcell有四個門，隱藏層單元是rnn的四倍
                              padding=self.padding,
                              bias=self.bias)       
    def branch1(self, x):
        x = self.conv(x)
        cc_i, cc_f, cc_o, cc_g = torch.split(x, self.hidden_dim, dim=1) 
        return cc_i, cc_f, cc_o, cc_g
    
    def forward(self, input_tensor, cur_state):
        #input_tensor的尺寸為（batch_size，channel，weight，width），沒有time_step
        #cur_state的尺寸是（batch_size,（hidden_dim）channel，weight，width），是呼叫函式init_hidden返回的細胞狀態
#        cnn1dOut=F.max_pool2d(self.out1x1(torch.cat(outputs, 1)),(2,2))  #
#        print('input_tensor'+str(input_tensor.shape))2
#        print('cur_state'+str(cur_state.shape))
        h_cur, c_cur = cur_state
#        h_cur=F.max_pool2d(h_cur,(2,2))
#        c_cur=F.max_pool2d(h_cur,(2,2))
#        print('1')        
#        print(input_tensor.shape)
#        print('2') 
#        print(h_cur.shape)      
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        #conv層的卷積不需要和linear一樣，可以是多維的，只要channel數目相同即可
#        print('combine shape'+str(combined.shape))
        
        
#         combined_conv = self.conv(combined)
# #        print('combined_conv shape'+str(combined_conv.shape))
#         cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        y = combined
        y = y + torch.zeros(1, dtype=y.dtype, device=y.device, requires_grad=True)       
        cc_i, cc_f, cc_o, cc_g=checkpoint(self.branch1, y)
        
        
        #使用split函式把輸出4*hidden_dim分割成四個門
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

#        print('f'+str(f.shape))
#        print('c_cur'+str(c_cur.shape))
        c_next = f * c_cur + i * g   #下一個細胞狀態
        h_next = o * torch.tanh(c_next)  #下一個hc

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
#        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
#                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))
        return (torch.zeros(batch_size, self.hidden_dim, height, width,device='cuda' ),
                torch.zeros(batch_size, self.hidden_dim, height, width,device='cuda'))

class ConvLSTM(nn.Module):

    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)
        #核對尺寸，用的函式是靜態方法

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        #kernel_size==hidden_dim=num_layer的維度，因為要遍歷num_layer次
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        #如果return_all_layers==true，則返回所有得到h，如果為false，則返回最後一層的最後一個h

        cell_list = []
        for i in range(0, self.num_layers):
            #判斷input_dim是否是第一層的第一個輸入，如果是的話則使用input_dim，否則取第i層的最後一個hidden_dim的channel數作為輸入
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        #以num_layer為三層為例，則cell_list列表裡的內容為[convlstmcell0（），convlstmcell1（），convlstmcell2（）]
        #Module_list把nn.module的方法作為列表存放進去，在forward的時候可以呼叫Module_list的東西，cell_list【0】，cell_list【1】，
        #一直到cell_list【num_layer】，
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        #第一次傳入hidden_state為none
        #input_tensor的size為（batch_size,time_step,channel,height,width）
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        #在forward裡開始構建模型，首先把input_tensor的維度調整，然後初始化隱藏狀態
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            #呼叫convlstm的init_hidden方法不是lstmcell的方法
            #返回的hidden_state有num_layer個hc，cc
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)#取time_step
        cur_layer_input = input_tensor

        #初始化h之後開始前向傳播
        for layer_idx in range(self.num_layers):
            #在已經初始化好了的hidden_state中取出第num_layer個狀態給num_layer的h0，c0，其作為第一個輸入
            h, c = hidden_state[layer_idx]
            output_inner = []
            #開始每一層的時間步傳播
            for t in range(seq_len):
                #用cell_list[i]表示第i層的convlstmcell，計算每個time_step的h和c
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                #將每一次的h存放在output_inner裡
                output_inner.append(h)
            #layer_output是五維向量，在dim=1的維度堆疊，和input_tensor的維度保持一致
            layer_output = torch.stack(output_inner, dim=1)
            #吧每一層輸出肚餓五維向量作為下一層的輸入，因為五維向量的輸入沒有num_layer，所以每一層的輸入都要喂入五維向量
            cur_layer_input = layer_output
            #layer_output_list存放的是第一層，第二層，第三層的每一層的五維向量，這些五維向量作為input_tensor的輸入
            layer_output_list.append(layer_output)
            #last_state_list裡面存放的是第一層，第二層，第三次最後time_step的h和c
            last_state_list.append([h, c])

        if not self.return_all_layers:
            #如果return_all_layers==false的話，則返回每一層最後的狀態，返回最後一層的五維向量，返回最後一層的h和c
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]


        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            #cell_list[i]是celllstm的單元，以呼叫裡面的方法
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
            #返回的init_states為num_layer個hc=（batch_size,channel(hidden_dim),height,width），cc=（batch_size,channel(hidden_dim),height,width）
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
#%%
model = ConvLSTM(input_dim=1,
         hidden_dim=[1, 1],
         kernel_size=(3, 3),
         num_layers=2,
         batch_first=True,
         bias=True,
         return_all_layers=False).cuda()


combinetest=torch.randn(1, 24,1, 70, 120).cuda() 
tt=model(combinetest)

