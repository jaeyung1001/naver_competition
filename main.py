import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report,precision_score,recall_score,accuracy_score
from keras.layers import *
from keras.models import *
from keras.utils import *
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.regularizers import l2
data = pd.read_csv('data.csv')
ans=pd.get_dummies(data['diagnosis'],drop_first=True)

data=pd.concat([data,ans],axis=1)
data.drop('diagnosis',axis=1,inplace=True)
data.drop('Unnamed: 32',axis=1,inplace=True)
data.drop('id',axis=1,inplace=True)
#data.drop(['texture_se','texture_mean','texture_worst','symmetry_se','smoothness_se'],axis=1,inplace=True)


X_train,X_test,y_train,y_test=train_test_split(data.drop('M',axis=1),data['M'],test_size=0.3)

input_dim = 30
DROP_OUT = 0.1
DENSE_DIM = 100
# Simple MLP model start


model = Sequential()

model.add(Dense(DENSE_DIM, input_dim=input_dim, activation='elu'))
model.add(Dropout(DROP_OUT))
model.add(BatchNormalization())
model.add(Dense(DENSE_DIM, activation='elu'))
model.add(Dropout(DROP_OUT))
#model.add(BatchNormalization())
#model.add(Dense(DENSE_DIM, activation='elu'))
#model.add(Dropout(DROP_OUT))
#model.add(BatchNormalization())
#model.add(Dense(DENSE_DIM, activation='elu'))
#model.add(Dropout(DROP_OUT))
model.add(Dense(1, activation='sigmoid'))



# file save, earlystopping, reduce learning rate
filepath=  "models/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

lr_sched = ReduceLROnPlateau(monitor='val_acc', factor=0.85, patience=20, cooldown=1, verbose=1)
early_stopping = EarlyStopping(monitor='val_acc', patience=30)


callbacks_list = [checkpoint, lr_sched, early_stopping]

# compile
optimizer = Adam(0.0001)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.fit(x = X_train, y = y_train,
	validation_data=(X_test,y_test),
	batch_size = 16,
	epochs = 1000,
	shuffle = True,
	callbacks = callbacks_list,
	verbose = 1)
print(model.summary())
preds = model.predict_classes(X_test, verbose=0)
#print(preds)
print(accuracy_score(y_test,preds))
