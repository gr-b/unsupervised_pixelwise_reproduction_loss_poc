import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.layers import Input, Dense, Dropout, Lambda, BatchNormalization, Activation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.keras import losses


import tensorflow as tf

from PIL import Image

def show(image):
	plt.imshow(image, cmap='gray')
	plt.show()


r = 25
batchSize = 1
w, h = 100, 100


# Generates an image of the given width, height
# With a randomly placed circle of random radius
def generate_circle_image(w, h, cx, cy):
	xgrid, ygrid = np.meshgrid(np.arange(0, w), np.arange(0, h))
	#cx, cy, cr = np.random.randint(0, w), np.random.randint(0, h), np.random.randint(0, int(w/3))
	# Stack grids so that they broadcast with xs, ys
	xgrid = np.stack([xgrid]*cx.shape[0])
	ygrid = np.stack([ygrid]*cy.shape[0])

	xcomp = ((xgrid.T - cx).T)**2
	ycomp = ((ygrid.T - cy).T)**2 
	
	circle = ((xcomp + ycomp).T / (2*(100**2))).T
	return circle

def generate_single_circle_image(w, h, cx, cy):
	xgrid, ygrid = np.meshgrid(np.arange(0, w), np.arange(0, h))
	circle = (xgrid-cx)**2 + (ygrid-cy)**2
	circle = (circle / (2*(100**2)))
	return circle


def generate_data(w, h, n):
	y = np.array([[np.random.randint(0, w), np.random.randint(0, h)] for i in range(n)])
	x = generate_circle_image(w, h, y[:, 0], y[:, 1])
	return x, y

n = 2000
x_train, y_train = generate_data(w, h, n)
#x_train = np.array([x_train[0]]*n)
#y_train = np.array([y_train[0]]*n)



xgrid, ygrid = np.meshgrid(np.arange(0, w), np.arange(0, h))
#xgrid = np.stack([xgrid]*batchSize)
#ygrid = np.stack([ygrid]*batchSize)


def pixelwise_reproduction_loss(y_true, y_pred):
	cx, cy = y_pred[:,0], y_pred[:,1]

	x_grid, y_grid = K.variable(value=xgrid), K.variable(value=ygrid)
	
	xcomp = (K.transpose(K.transpose(x_grid) - cx))**2
	ycomp = (K.transpose(K.transpose(y_grid) - cy))**2 

	circle = K.transpose(K.transpose(xcomp + ycomp))
	circle = (circle / (2*(100**2)))
	
	circle = K.reshape(circle, (w*h, ))

	mse = K.mean(((circle-y_true)**2))
	return mse
	
	

def model():
	model = Sequential()
	model.add(Dense(768, input_shape=(w*h,), activation="relu"))
	model.add(Dropout(0.2))	

	model.add(Dense(256, activation='relu'))
	model.add(Dense(128, activation='relu'))	
	model.add(Dense(32, activation='relu'))
	model.add(Dense(2, activation='relu'))
	# output is (x, y, r)
	model.compile(loss=pixelwise_reproduction_loss, optimizer='adam')
	return model

model = model()
print(model.summary())

x_train = x_train.reshape(n, w*h)
model.fit(x_train, x_train, epochs=1, verbose=1, batch_size=batchSize)


#x_test, y_test = generate_data(w, h, n)
#x_test = x_test.reshape(n, w*h)
y_pred = model.predict(x_train)

mse = np.mean((y_pred-y_train)**2)

print("MSE:" + str(mse))

plt.imshow(x_train[0].reshape(w, h), cmap='gray')
plt.show()

y = y_pred[0]
y_img = generate_single_circle_image(w, h, y[0], y[1])

show(y_img)


