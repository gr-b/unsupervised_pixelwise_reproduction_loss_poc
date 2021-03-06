import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.models import load_model

from PIL import Image



w, h = 100, 100

image = Image.open('input.png')
image = image.resize((w, h))
image = np.array(image)

def toGrayscale(rgb):
	r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
	return 0.2989 * r + 0.5870 * g + 0.1140 * b

grayimg = toGrayscale(image)
#plt.imshow(grayimg, cmap='gray')
#plt.show()
#print(grayimg.shape)


# Generates an image of the given width, height
# With a randomly placed circle of random radius
def generate_circle_image(w, h, cx, cy, cr):
	xgrid, ygrid = np.meshgrid(np.arange(0, w), np.arange(0, h))
	#cx, cy, cr = np.random.randint(0, w), np.random.randint(0, h), np.random.randint(0, int(w/3))
	# Stack grids so that they broadcast with xs, ys
	xgrid = np.stack([xgrid]*cx.shape[0])
	ygrid = np.stack([ygrid]*cy.shape[0])

	xcomp = ((xgrid.T - cx).T)**2
	ycomp = ((ygrid.T - cy).T)**2 
	
	circle = ((xcomp + ycomp).T < cr**2).T
	return circle

def generate_single_circle_image(w, h, cx, cy, cr):
	xgrid, ygrid = np.meshgrid(np.arange(0, w), np.arange(0, h))
	circle = (xgrid-cx)**2 + (ygrid-cy)**2 < cr**2
	return circle


def generate_data(w, h, n):
	y = np.array([[np.random.randint(0, w), np.random.randint(0, h), np.random.randint(0, int(w/3))] for i in range(n)])
	x = generate_circle_image(w, h, y[:, 0], y[:, 1], y[:, 2])
	
	return x, y

n = 200
x_train, y_train = generate_data(w, h, n)


def model():
	model = Sequential()
	model.add(Dense(1024, input_shape=(w*h,), activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(64, activation='relu'))
	
	model.add(Dense(3, activation='relu'))
	# output is (x, y, r)
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

model = model()

x_train = x_train.reshape(n, w*h)
model.fit(x_train, y_train, epochs=10, verbose=1, batch_size=32)


x_test, y_test = generate_data(w, h, n)
x_test = x_test.reshape(n, w*h)
y_pred = model.predict(x_test)

mse = np.mean((y_pred-y_test)**2)

print("MSE:" + str(mse))

plt.imshow(x_test[0].reshape(w, h), cmap='gray')
plt.show()
y = y_test[0]
y_img = generate_single_circle_image(w, h, y[0], y[1], y[2])
print(y_img.shape)
plt.imshow(y_img, cmap='gray')
plt.show()


