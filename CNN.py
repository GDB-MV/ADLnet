#loading drive which contains the files
from google.colab import drive
drive.mount('/content/gdrive')



#code to set Google GPU as default
%tensorflow_version 2.x
import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))



#loading the files
import numpy as np
import math

window = 60
dim_output = 10
all_X = np.loadtxt("")  #here has to be inserted the directory of the training/validation instances
all_y = np.loadtxt("")  #here has to be inserted the directory of the training/validation associated ground truth
# check on the training/validation dataset
A = all_X.shape
B = all_y.shape
if A[0] == B[0]:
  N = A[0]
print("N: ", N)
X = np.empty((N, window, 1))
y_true = np.empty((N, dim_output))
for i in range(N):
  X[i,:, 0] = all_X[i,:]
  y_true[i, :] = all_y[i,:]
  
  
  
#dataset training and evaluation on the training dataset
import tensorflow as tf
import time
from keras import backend as K

window = 60
dim_output = 10

def recall(y_true, y_pred):
  true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
  recall = true_positives / (possible_positives + K.epsilon())
  return recall

def precision(y_true, y_pred):
  true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
  precision = true_positives / (predicted_positives + K.epsilon())
  return precision

def f1(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2*((prec*rec)/(prec+rec+K.epsilon()))

tf.keras.backend.clear_session()

model = tf.keras.Sequential([
    tf.keras.layers.BatchNormalization(input_shape=(window, 1)),   
    tf.keras.layers.Conv1D(256, 8, activation='relu'),
    tf.keras.layers.Conv1D(128, 6, activation='relu'),
    tf.keras.layers.Conv1D(128, 6, activation='relu'),   
    tf.keras.layers.Conv1D(64, 4, activation='relu'),
    tf.keras.layers.Conv1D(64, 4, activation='relu'),
    tf.keras.layers.Conv1D(32, 2, activation='relu'),
    tf.keras.layers.Conv1D(32, 2, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0,l2=0.1)),
    tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0,l2=0.1)),
    tf.keras.layers.Dense(8, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0,l2=0.1)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(.5),
    tf.keras.layers.Dense(dim_output, activation = 'softmax')
    ])

model.compile(loss='categorical_crossentropy', optimizer='ADAM', metrics=['accuracy', precision, recall, f1])

print(model.summary())

history = model.fit(X, y_true, epochs=100, verbose=1, validation_split=0.3, batch_size=512)

loss, accuracy, f1_score, precision, recall = model.evaluate(X, y_true, verbose=0)
print("N: ", N)
print("Loss: ", loss)
print("Accuracy: ", accuracy)
print("F1 Score: ", f1_score)
print("Precision: ", precision)
print("Recall: ", recall)



#evaluation of the model on the testing dataset
all_X = np.loadtxt("") #here has to be inserted the directory of the testing instances
all_y = np.loadtxt("") #here has to be inserted the directory of the testing associated ground truth
# check on the testing dataset
A = all_X.shape
B = all_y.shape
if A[0] == B[0]:
  N = A[0]
print("N: ", N)
X = np.empty((N, window, 1))
y_true = np.empty((N, dim_output))
for i in range(N):
  X[i,:, 0] = all_X[i,:]
  y_true[i, :] = all_y[i,:]

loss, accuracy, f1_score, precision, recall = model.evaluate(X, y_true, verbose=0)
print("Loss: ", loss)
print("Accuracy: ", accuracy)
print("F1 Score: ", f1_score)
print("Precision: ", precision)
print("Recall: ", recall)



#plot the training and validation history
import matplotlib.pyplot as plt

metric = 'loss'
yscale = 'linear'
train_metrics = history.history[metric]
val_metrics = history.history['val_'+metric]
start_epoch = 1
epochs = range(start_epoch, len(train_metrics) + 1)

plt.figure(figsize=(60, 5))
plt.subplot(121)
plt.grid(True)
plt.yscale(yscale)
plt.plot(epochs, train_metrics, epochs, val_metrics)
plt.xticks(np.arange(start_epoch, len(train_metrics) + 1, 1.0))
plt.xlabel("Epochs")
plt.ylabel(metric)
plt.ylim([0.1, 0.8])
plt.title('Training and validation '+ metric)
plt.legend(["train_"+metric, 'val_'+ metric])
