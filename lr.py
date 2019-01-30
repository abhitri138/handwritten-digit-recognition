import numpy as np
from struct import unpack
from sklearn.linear_model import LogisticRegression

def read(image_file, label_file):
    train_images = open(image_file , 'rb')
    train_labels = open(label_file, 'rb')

    train_images.read(4)
    num_images = unpack('>I', train_images.read(4))[0]
    num_rows = unpack('>I', train_images.read(4))[0]
    num_cols = unpack('>I', train_images.read(4))[0]

    train_labels.read(4)
    num_labels = unpack('>I', train_labels.read(4))[0]

    x = np.zeros((num_labels, num_rows*num_rows), dtype=np.uint8)
    y = np.zeros(num_labels, dtype=np.uint8)
    for i in range(num_labels):
        for j in range(num_rows*num_cols):
            x[i][j] = unpack('>B', train_images.read(1))[0]
        y[i] = unpack('>B', train_labels.read(1))[0]
    train_images.close()
    train_labels.close()
    return (x, y)
    
x_train, y_train = read('train-images.idx3-ubyte', 'train-labels.idx1-ubyte')
x_test, y_test = read('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte')

lr = LogisticRegression(solver = 'lbfgs')
lr.fit(x_train, y_train)
print('Done training')

score = lr.score(x_test, y_test)
print(score)

pred_labels = lr.predict(x_test)

array = np.zeros((pred_labels.shape[0], 10))
array[np.arange(pred_labels.shape[0]), pred_labels] = 1

with open("lr.csv", "wb") as f:
    np.savetxt(f, arr.astype(int), fmt='%i', delimiter=",")

print('CSV Generated')