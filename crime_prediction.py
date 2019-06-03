import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import warnings

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

warnings.filterwarnings('ignore')
data = pd.read_csv('crimes.csv')

#print(data.head())
#print(data['Primary Type'].unique())
#print(data.dtypes)

X = data.iloc[:,[5,7,11,12,13,18]].values
Y = data.iloc[:,6].values.reshape(-1,1)

encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(Y)

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
X[:,0] = le.fit_transform(X[:,0])
X[:,1] = le.fit_transform(X[:,1])

xtrain, xtest, ytrain, ytest = train_test_split(X , y , test_size=0.30 , random_state=0)

model = Sequential()
model.add(Dense(32, input_shape=(6,), activation='relu', name='fc1'))
model.add(Dense(32, activation='relu', name='fc2'))
model.add(Dense(32, activation='relu', name='fc3'))
model.add(Dense(64, activation='relu', name='fc4'))
model.add(Dense(64, activation='sigmoid', name='fc5'))
model.add(Dense(32, activation='sigmoid', name='fc6'))
model.add(Dense(33, activation='softmax', name = 'output'))
optimizer = Adam(lr=0.001)
model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
print('Neural Network Model Summary: ')
print(model.summary())

model.fit(xtrain, ytrain, verbose=2, batch_size=10000, epochs=30)

results = model.evaluate(xtest, ytest)
print('Final test set loss: {:4f}'.format(results[0]))
print('Final test set accuracy: {:4f}'.format(results[1]))
