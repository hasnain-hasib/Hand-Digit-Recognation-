import tensorflow as tf
import matplotlib.pyplot as plt
import  numpy as np
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test)= mnist.load_data()


x_train = tf.keras.utils.normalize(x_train, axis =1)
x_test = tf.keras.utils.normalize(x_test,axis =1 )


model = tf.keras.Sequential([

    tf.keras.layers.Flatten(input_shape=(28,28,1)),
    tf.keras.layers.Dense(128,activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation= tf.nn.softmax)

])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x=x_train, y=y_train, epochs =10)

test_loss, test_acc = model.evaluate ( x= x_test, y= y_test)

print('\nTest Accuracy : ', test_acc)

img = x_test[0]
img = np.array([img])
predictions = model.predict(img)

print (np.argmax(predictions))

plt.imshow(x_test[0],cmap="gray")
plt.show()

