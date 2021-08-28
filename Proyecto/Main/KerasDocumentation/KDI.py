from tensorflow.keras.models import Sequential

model = Sequential()

#Stacking layers is as easy as .add()

from tensorflow.keras.layers import Dense

model.add(Dense(units = 64, activation = 'relu'))
model.add(Dense(units=10, activation='softmax'))

#Configure its learning process with .compile()
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
