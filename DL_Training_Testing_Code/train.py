import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras import backend
from keras.models import model_from_json
from keras.layers import Conv2D,Conv2DTranspose,Conv3D,Conv3DTranspose
from keras.layers import Input,add,LeakyReLU,ReLU,Dropout,BatchNormalization,Activation,concatenate,Dense

Echo_time=18
fnum=19
bins=6
step=4
maxR=40

basepath='../Sample_Data/'
trainpath=basepath+'S_S0_40_real_train.csv'
testpath=basepath+'S_S0_40_real_test.csv'

#Read the signal (s/s0) and cbv from the csv file for training
combined_arr = np.array(pd.read_csv(trainpath,header=None))
np.take(combined_arr,np.random.permutation(combined_arr.shape[0]),axis=0,out=combined_arr) #Use only for S_S0_40_hybrid14 
combined_arr = combined_arr[combined_arr[:,Echo_time+maxR+2]<40]
ssize=combined_arr.shape
total_samples = ssize[0]
X_train = combined_arr[:,0:Echo_time]
vsd_arr_tmp = combined_arr[:,Echo_time:Echo_time+maxR]
cbvarr = combined_arr[:,Echo_time+maxR+2]+1#/100.0
cbvarr = np.reshape(cbvarr,(ssize[0],1))#/25

print('Min and Max cbv: ',np.min(cbvarr),np.max(cbvarr))

#normalize the VSD by dividing with the maximum value
for i in range(total_samples):
    vsd_arr_tmp[i]/=np.max(vsd_arr_tmp[i])

# create the ground truth output by concatenating the VSD and CBV values 
y_train = np.concatenate((vsd_arr_tmp,cbvarr),axis=1)
print(X_train.shape,y_train.shape)

#Read the test file and create the feature set (X_test) and output (Y_test)
combined_arr1 = np.array(pd.read_csv(testpath,header=None))
combined_arr1 = combined_arr1[combined_arr1[:,Echo_time+maxR+2]<40]
ssize1=combined_arr1.shape
total_samples1 = ssize1[0]
X_test = combined_arr1[:,0:Echo_time]
vsd_arr_tmp1 = combined_arr1[:,Echo_time:Echo_time+maxR]
cbvarr1 = combined_arr1[:,Echo_time+maxR+2]+1
cbvarr1 = np.reshape(cbvarr1,(ssize1[0],1))
for i in range(total_samples1):
    vsd_arr_tmp1[i]/=np.max(vsd_arr_tmp1[i])

y_test = np.concatenate((vsd_arr_tmp1,cbvarr1),axis=1)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

def build_cyclic_NN(ip_shape,op_dim):
    input = Input(shape=ip_shape, name='input')
    stack1 = keras.Sequential([
        keras.layers.Dense(units=2048, activation='relu'), #hidden layer 1
        keras.layers.Dense(units=1024, activation='relu'), #hidden layer 1
        keras.layers.Dense(units=512, activation='relu'), #hidden layer 2
        keras.layers.Dense(units=256, activation='relu'), #hidden layer 2
        keras.layers.Dense(units=128, activation='relu'), #hidden layer 2
        keras.layers.Dense(units=64, activation='relu'), #hidden layer 2
    ]) #Network architecture of the CBVE
    intermediate =stack1(input)    
    CBV_out = keras.layers.Dense(units=16, activation='relu')(intermediate)
    CBV_out = keras.layers.Dense(units=8, activation='relu')(CBV_out)
    CBV_out = keras.layers.Dense(units=1, activation='relu')(CBV_out)

    CBV_Model = keras.Model(inputs=input,outputs=CBV_out)
    CBV_Model.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss=['mse'], metrics='mse') #Working so far
    CBV_Model.trainable=False #First train CBVE and then set the weights tobe non-trainable
    CBV_pred = CBV_Model(input) 

    recon_concat1=concatenate([input,CBV_pred],axis=1) #Output of the CBVE is concatenated with the input and passed to VSDE

    stack2 = keras.Sequential([
        keras.layers.Dense(units=2048, activation='relu'), #hidden layer 1 (Originally 2048)
        keras.layers.Dense(units=1024, activation='relu'), #hidden layer 1 (Originally 1024)
        keras.layers.Dense(units=1024, activation='relu'), #hidden layer 1 (Originally not needed)
        keras.layers.Dense(units=512, activation='relu'), #hidden layer 2 (Original Softplus)
        keras.layers.Dense(units=256, activation='relu'), #hidden layer 2 (Original softplus)
        keras.layers.Dense(units=128, activation='relu'), #hidden layer 2 (Original softplus)
        keras.layers.Dense(units=124, activation='relu'), #hidden layer 2 (Original is 64 and softplus)
    ]) #Network architecture of the VSDE

    Reverse_layer = stack2(recon_concat1)
    VSD_out = keras.layers.Dense(units=op_dim, activation='sigmoid')(Reverse_layer)
    VSD_Model = keras.Model(inputs=input,outputs=[VSD_out])
    VSD_Model.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss=['mse'], metrics='mse') #Working so far
    CBV_Model.trainable=True

    return CBV_Model,VSD_Model


#save the trained model

def save_model(model,modeljson,modelweights):
    model_json = model.to_json() 
    with open(modeljson, "w") as json_file:
        json_file.write(model_json)
    model.save_weights(modelweights)
    print("Model Saved")
    print('--------------------------------------------------------------------')

# load the trained model

def load_model(modeljson,modelweights):
    json_file = open(modeljson, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    print('Model Json Loaded')
    model.load_weights(modelweights)
    print("Model Weights Loaded")
    print('--------------------------------------------------------------------')
    return model

CBVMod,VSDMod=build_cyclic_NN((fnum-1,),maxR)
print(CBVMod.summary())
print(VSDMod.summary())

#Train the CBV and VSD models
history = CBVMod.fit(x=X_train, y=y_train[:,maxR], batch_size=32, epochs=15, validation_data=(X_test,y_test[:,maxR]),verbose=1)
history = VSDMod.fit(x=X_train, y=y_train[:,0:maxR], batch_size=32, epochs=40, validation_data=(X_test,y_test[:,0:maxR]),verbose=1)
print("Training CBVE.......")
save_model(CBVMod,basepath+'Cbv_predictor_14k_TL_3.json',basepath+'Weights_Cbv_predictor_14k_TL_3.h5')
print("Training VSDE.......")
save_model(VSDMod,basepath+'VSD_predictor_14k_TL_3.json',basepath+'Weights_VSD_predictor_14k_TL_3.h5')

#Test the trained model on the test set
cbv_prediction = CBVMod.predict(X_test)
prediction2 = VSDMod.predict(X_test)

#Write true and predicted CBV in a CSV file
file=open(basepath+'CBV_real_train_test_WB.csv','w')
for i in range(len(cbv_prediction)):
    file.write(str(y_test[i][maxR])+','+str(cbv_prediction[i][0])+'\n')
file.close()

#Plot all the predicted VSDs
offset=0
x_axis = np.linspace(0,maxR,maxR)
for i in range(0,60):
    y_predict=prediction2[i+offset]/np.max(prediction2[i+offset])
    plt.plot(x_axis,y_predict)
plt.savefig(basepath+'Pred_curve_real_train_WB'+'.png')

#Plot true and predicted VSDs
fig, axs = plt.subplots(nrows=12,ncols=5,figsize=(18, 15))
fact=0
row = 0
col=0
x_axis = np.linspace(0,maxR,maxR)
for i in range(0,60):
    if int(i/5)>fact:
        fact=int(i/5)
        row=row+1
        col=0
    y_true=y_test[i+offset][0:maxR]
    y_predict=prediction2[i+offset]/np.max(prediction2[i+offset])
    axs[row][col].plot(x_axis,y_true)
    axs[row][col].plot(x_axis,y_predict)
    axs[row][col].legend(['true('+str(np.round(y_test[i+offset][maxR],1))+')', 'pred('+str(np.round(cbv_prediction[i+offset],1))+')'], loc='upper right')
    col=col+1
plt.show()
plt.savefig(basepath+'Test_real_train_WB'+str(offset)+'.png')
