

import os
import numpy as np

from sklearn.utils import shuffle
import datetime
# import numba as nb
#from pytorchtools import EarlyStopping
import numpy as np
from InceptionConvlstm import*
torch.backends.cudnn.benchmark = True
#%%
data=np.load('dataf1~f24(2017).npy')

def getTestindex(input_timestep=72,skip_hours=0):
    test_time=['17011501','17011601','17011701','17011801','17011901',
           '17021501','17021601','17021701','17021801','17021901',
           '17031501','17031601','17031701','17031801','17031901',
           '17041501','17041601','17041701','17041801','17041901',
           '17051501','17051601','17051701','17051801','17051901',
           '17061501','17061601','17061701','17061801','17061901',
           '17071501','17071601','17071701','17071801','17071901',
           '17081501','17081601','17081701','17081801','17081901',
           '17091501','17091601','17091701','17091801','17091901',
           '17101501','17101601','17101701','17101801','17101901',
           '17111501','17111601','17111701','17111801','17111901',
           '17121501','17121601','17121701','17121801','17121901',
           '18011501','18011601','18011701','18011801','18011901',
           '18021501','18021601','18021701','18021801','18021901',
           '18031501','18031601','18031701','18031801','18031901',
           '18041501','18041601','18041701','18041801','18041901',
           '18051501','18051601','18051701','18051801','18051901',
           '18061501','18061601','18061701','18061801','18061901',
           '18071501','18071601','18071701','18071801','18071901',
           '18081501','18081601','18081701','18081801','18081901',
           '18091501','18091601','18091701','18091801','18091901',
           '18101501','18101601','18101701','18101801','18101901',
           '18111501','18111601','18111701','18111801','18111901',
           '18121501','18121601','18121701','18121801','18121901',
           '19011501','19011601','19011701','19011801','19011901',
           '19021501','19021601','19021701','19021801','19021901',
           '19031501','19031601','19031701','19031801','19031901',
           '19041501','19041601','19041701','19041801','19041901',
           '19051501','19051601','19051701','19051801','19051901',
           '19061501','19061601','19061701','19061801','19061901',
           '19071501','19071601','19071701','19071801','19071901',
           '19081501','19081601','19081701','19081801','19081901',
           '19091501','19091601','19091701','19091801','19091901',
           '19101501','19101601','19101701','19101801','19101901',
           '19111501','19111601','19111701','19111801','19111901',
           '19121501','19121601','19121701','19121801','19121901']

    date_range=[]
    start_time = datetime.datetime(2017,1,1,1)
    end_time = datetime.datetime(2017,12,31,23)     
    d = start_time
    delta = datetime.timedelta(hours=1)
    while d<=end_time:           
        format_time = d.strftime("17%m%d%H")
        date_range.append(format_time)
        d = d+delta

    start_time = datetime.datetime(2018,1,1,00)
    end_time = datetime.datetime(2018,12,31,23)     
    d = start_time
    delta = datetime.timedelta(hours=1)
    while d<=end_time:           
        format_time = d.strftime("18%m%d%H")
        date_range.append(format_time)
        d = d+delta

    start_time = datetime.datetime(2019,1,1,00)
    end_time = datetime.datetime(2019,12,31,23)     
    d = start_time
    delta = datetime.timedelta(hours=1)
    while d<=end_time:           
        format_time = d.strftime("19%m%d%H")
        date_range.append(format_time)
        d = d+delta

    y_start_time = date_range[input_timestep+skip_hours::]
    
    test_index = []
    for a in test_time:
        test_index.append(y_start_time.index(a)) 
    return test_index



def zerotemp(datalen=26280,input_timestep=72):
    
    sample = datalen-input_timestep
    Error_input = np.zeros(shape=(sample,input_timestep,70,120), dtype=np.float16)
    
    return Error_input,sample
    



def getTimeseries(Error_input,sample,input_timestep=72):
    global data
    for index in range(0,sample):
        Error_input[index] = data[index:index+input_timestep]
    
    Error_input_reshape = np.reshape(Error_input,(Error_input.shape[0],
                                              Error_input.shape[1],
                                              Error_input.shape[2],
                                              Error_input.shape[3],1))
    return Error_input_reshape

    
test_index=getTestindex()
input_error,sample=zerotemp()
input_error=getTimeseries(input_error,sample)
#del data



 
input_timestep=72
skip_hours=0

i=0



  
Error_output = data[(input_timestep+i+skip_hours)::] #y
#    Error_input_reshape_=Error_input_reshape[:len(Error_output)]




input_error_=input_error[:len(Error_output)]



testX = input_error_[test_index]
testy = Error_output[test_index]


index=[i for i in range(len(input_error_))]
index=np.array(index)
input_error_index = np.delete(index,test_index,0)

input_error_ = input_error_[input_error_index]


index=[i for i in range(len(Error_output))]
index=np.array(index)
Error_output_index = np.delete(index,test_index,0)
Error_output = Error_output[Error_output_index]


trainX = input_error_

trainy = Error_output


#early_stopping = EarlyStopping(monitor='loss', patience=3, verbose=1) 
del input_error,input_error_,data
from sklearn.model_selection import train_test_split
trainX, valX, trainy, valy = train_test_split(trainX, trainy, test_size=0.3, random_state=42)



trainX=np.reshape(trainX,(trainX.shape[0],trainX.shape[1],trainX.shape[4],trainX.shape[2],trainX.shape[3]))

valX=np.reshape(valX,(valX.shape[0],valX.shape[1],valX.shape[4],valX.shape[2],valX.shape[3]))


trainy=np.reshape(trainy,(trainy.shape[0],trainy.shape[1],trainy.shape[2]))
valy=np.reshape(valy,(valy.shape[0],valy.shape[1],valy.shape[2]))

#trainX=trainX.astype('float32')
#valX=valX.astype('float32')
#
#trainy=trainy.astype('float32')
#valy=valy.astype('float32')

#%%


model = ConvLSTM(input_dim=1,
                 hidden_dim=[32, 1],
                 kernel_size=(3, 3),
                 num_layers=2,
                 batch_first=True,
                 bias=True,
                 return_all_layers=False)

model=model.cuda()
#model=model
print(model)
#%%
x=torch.from_numpy(trainX[0:1]).float().cuda()

output = model2(x)
print(output.shape)
target=torch.from_numpy(trainy[0:1]).float().cuda()
target.shape
loss.backward()
loss = criterion(output, target)

test,_=model(x)
test=test[0][:,-1,:]

test.shape
#%%

model2=InceptionConvLstmNetwork(hidden_dim=[32, 1])
#%%
model2=InceptionConvLstmDenseNetwork(hidden_dim=[32, 1])
model2=model2.cuda()

#%%
epoch=0
e_sp = time.time()
total_loss = 0
compution_time = 0
for i in range(3):
        data = x
        target = target

        b_sp = time.time()
        output = model2(data)
        print('1')
#            print(output[0].shape)
        loss = criterion(output, target)
#            loss=loss.float() 
        print('2')
        optimizer.zero_grad()
        print('3')
        loss.backward()
        print('4')
        optimizer.step()
        print('5')
        compution_time += time.time() - b_sp
        print('6')
        total_loss+=float(loss)
        print('7')
        torch.cuda.empty_cache()
        # optimizer.step()
print('epoch end')
epoch_time = time.time() - e_sp

#combinetest=torch.randn(1, 65, 70, 120)#combined_conv shapetorch.Size([1, 4, 70, 120])

#%%
testX=np.reshape(testX,(testX.shape[0],testX.shape[1],testX.shape[4],testX.shape[2],testX.shape[3]))

testX=torch.from_numpy(testX).float().cuda()
testy=torch.from_numpy(testy).float().cuda()
model2.eval()
testX.shape
x.shape
with torch.no_grad():
    
    output = model2(testX)
    loss = criterion(output, testy)
    print(loss.data)
#test2=model2(x)
#test2.shape
#%%

class InceptionConvLstmNetwork(nn.Module):
    def __init__(self, input_dim=1,hidden_dim=[8, 1],kernel_size=(3, 3),batch_first=True,return_all_layers=False):
        super(InceptionConvLstmNetwork, self).__init__()
        self.num_layers=len(hidden_dim)
        self.InceptionConvLstmBlock = ConvLSTM(input_dim=input_dim,
                 hidden_dim=hidden_dim,
                 kernel_size=kernel_size,
                 num_layers=self.num_layers,
                 batch_first=batch_first,
                 bias=True,
                 return_all_layers=return_all_layers)
        
        
        self.outLayer =  nn.Sequential(
         nn.AvgPool2d((4, 4)),
          nn.Conv2d(in_channels=1,out_channels=70*120,kernel_size=(17, 30))
        )

    def forward(self, input):
        out,_ = self.InceptionConvLstmBlock(input)
        out = self.outLayer(out[0][:,-1,:])
        out=out.view(-1,70,120)
        return out
#%%

class InceptionConvLstmDenseNetwork(nn.Module):
    def __init__(self, input_dim=1,hidden_dim=[8, 1],kernel_size=(3, 3),batch_first=True,return_all_layers=False):
        super(InceptionConvLstmDenseNetwork, self).__init__()
        self.num_layers=len(hidden_dim)
        self.InceptionConvLstmBlock = ConvLSTM(input_dim=input_dim,
                 hidden_dim=hidden_dim,
                 kernel_size=kernel_size,
                 num_layers=self.num_layers,
                 batch_first=batch_first,
                 bias=True,
                 return_all_layers=return_all_layers)

        self.outLayer =torch.nn.Linear(70*120,70*120)

    def forward(self, input):
        out,_ = self.InceptionConvLstmBlock(input)
        out=out[0][:,-1,:]
        out=out.view(out.size(0),-1)
        out = self.outLayer(out)
        out=out.view(-1,70,120)
        return out
        
#%%
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
import time
import torch.utils.data as Data


scaler = GradScaler(enabled=True)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model2.parameters(), lr=0.001)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model2.parameters(), lr=0.001)



#%%
#t=trainX[0,-24:,:,:,:]
trainX=trainX[:,-24:]
#trainy=trainy[:,-24:]
#%%
torch_dataset = Data.TensorDataset(torch.from_numpy(trainX).float()  , torch.from_numpy(trainy).float() )

loader = Data.DataLoader(
    dataset = torch_dataset,
    batch_size = 4,
    shuffle=True,  #采用两个进程来提取
    pin_memory=True, num_workers=1
)
#


#loader = Data.DataLoader(dataset=torch_dataset, batch_size=2, num_workers=0)
#model.half()
#%%

for epoch in range(1,21):
    e_sp = time.time()
    total_loss = 0
    compution_time = 0
    for batch_idx, (data, target) in enumerate(loader):
    
            data = data.cuda()
            target = target.cuda()
    
            b_sp = time.time()
            output = model2(data)
            print('1')
#            print(output[0].shape)
            loss = criterion(output, target)
            print('2')            
#            loss=loss.float() 
            optimizer.zero_grad()
            loss.backward()
            print('3')            
            optimizer.step()
            print('4')
            compution_time += time.time() - b_sp
            total_loss+=float(loss)
            print('5')
            torch.cuda.empty_cache()
            # optimizer.step()
    print('epoch end')
    epoch_time = time.time() - e_sp
    print('Train Epoch: {} \t Loss: {:.6f}\t epoch time: {:.6f} s\t epoch compution time: {:.6f} s'.format(
        epoch, total_loss / len(loader), epoch_time, compution_time))
    #combinetest=torch.randn(1, 65, 70, 120)#combined_conv shapetorch.Size([1, 4, 70, 120])
#test=model(x)

#testtt=test[0][0]
#testtt[:,-1,:].shape
    #%% half版!!!!!!!!!!

    
early_stopping = EarlyStopping(patience=3, verbose=True)    
    
for epoch in range(1,50):
    e_sp = time.time()
    total_loss = 0
    compution_time = 0
    for batch_idx, (data, target) in enumerate(loader):

            data = data.cuda()
            target = target.cuda()
            b_sp = time.time()

#            loss=loss.float() 
            optimizer.zero_grad()
          
            with autocast():
                output = model2(data)
        #            print(output[0].shape)

                loss = criterion(output, target)

                
            scaler.scale(loss).backward()

        #  scaler.scale(optimizer)
            scaler.step(optimizer)
          
            scaler.update()    
         
            compution_time += time.time() - b_sp
          
            total_loss+=float(loss)
          
            torch.cuda.empty_cache()
            # optimizer.step()
    print('epoch end')
    epoch_time = time.time() - e_sp
    print('Train Epoch: {} \t Loss: {:.6f}\t epoch time: {:.6f} s\t epoch compution time: {:.6f} s'.format(
        epoch, total_loss / len(loader), epoch_time, compution_time))    
#%%
scaler = GradScaler(enabled=True)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model2.parameters(), lr=0.001)
bachsize=1
epochs=50


#%%
i=0
for e in range(epochs):
    print(len(trainX))

    for b in range(0,len(trainX)-bachsize):

        optimizer.zero_grad() # if don't call zero_grad, the grad of each batch will be accumulated  
        x=torch.from_numpy(trainX[b:b+bachsize]).cuda()  
        y=torch.from_numpy(trainy[b:b+bachsize]).cuda()  
        
        with autocast():
            out,state = model(x)
            loss = criterion(out[0][:,-1,:], y)
            
            loss=loss.cuda()
            
        scaler.scale(loss).backward()
    #  scaler.scale(optimizer)
        scaler.step(optimizer)
        scaler.update()
                 
#        if b%2==0:
#            print('epoch: {}, batch: {}, loss: {}'.format(e + 1, b , loss.data))
    
    
    
    

    print('epoch: {}, batch: {}, loss: {}'.format(e + 1, b, loss.data))

#%%
#x=torch.from_numpy(trainX[0:1]).cuda()    
#out,state = model(x)
#print(out[0].shape)
from torch.autograd import Variable
#%%
#scaler = GradScaler()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
bachsize=1
epochs=100
i=0
for e in range(epochs):
    print(len(trainX))

    for b in range(0,len(trainX)-bachsize):
#        tot_loss =0.0
        x=torch.from_numpy(trainX[b:b+bachsize]).cuda()  
        y=torch.from_numpy(trainy[b:b+bachsize]).cuda()  
        optimizer.zero_grad() # if don't call zero_grad, the grad of each batch will be accumulated 
        
        with autocast():
            
            out,state = model(x)
#            print(out.shape)
            loss = criterion(out[0][:,-1,:], y)
            loss=loss.cuda()
#            tot_loss += loss.data
        scaler.scale(loss).backward()
    #  scaler.scale(optimizer)
        scaler.step(optimizer)
        scaler.update()


        if b % 100 == 0:    # print every 2000 mini-batches
                print('epoch: {}, batch: {}, loss: {}'.format(e + 1, b, loss.data)) 
#        if(i+1)%100==0:
            

#    print('epoch: {}, batch: {}, loss: {}'.format(e + 1, b, loss.data))
#%%
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
bachsize=1
epochs=200
i=0
for e in range(epochs):
    print(len(trainX))
    running_loss = 0.0
    for b in range(0,4):
        optimizer.zero_grad() # if don't call zero_grad, the grad of each batch will be accumulated          
#        x=torch.from_numpy(trainX[b:b+bachsize])  
#        y=torch.from_numpy(trainy[b:b+bachsize]) 
        x=Variable(trainX[b:b+bachsize].type(torch.cuda.FloatTensor))
        y=Variable(trainy[b:b+bachsize].type(torch.cuda.FloatTensor)  )     

        out,state = model(x)
        loss = criterion(out[0][:,-1,:], y)
        loss=loss
            
        loss.backward()
    #  scaler.scale(optimizer)
        optimizer.step()


#        if b % 2 == 0:    # print every 2000 mini-batches
#            print('epoch: {}, batch: {}, loss: {}'.format(e + 1, b, loss.data))
#            running_loss = 0.0
#    
    
    out,state = model(valX)
    loss = criterion(out[0][:,-1,:], valy)
    print('epoch: {}, batch: {}, loss: {}'.format(e + 1, b, loss.data))
  #%%
#scaler = GradScaler()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
bachsize=1
epochs=20
i=0
for e in range(epochs):
    print(len(trainX))
    tot_loss =0.0
    for b in range(0,100):

        x=torch.from_numpy(trainX[b:b+bachsize]).cuda()  
        y=torch.from_numpy(trainy[b:b+bachsize]).cuda()  
        optimizer.zero_grad() # if don't call zero_grad, the grad of each batch will be accumulated 
        
        with autocast():
            
            out,state = model(x)
#            print(out.shape)
            loss = criterion(out[0][:,-1,:], y)
            loss=loss.cuda()
#            tot_loss += loss.data
        scaler.scale(loss).backward()
    #  scaler.scale(optimizer)
        scaler.step(optimizer)
        scaler.update()
        tot_loss+=loss


    print('epoch: {}, batch: {}, loss: {}'.format(e + 1, b, tot_loss/100)) 
#        if(i+1)%100==0:
            

#    print('epoch: {}, batch: {}, loss: {}'.format(e + 1, b, loss.data))  