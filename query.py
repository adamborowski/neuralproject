import os

from keras.models import model_from_yaml

from modules import input_test

os.environ['THEANO_FLAGS'] = "device=gpu,floatX=float32"

import numpy as np

model = model_from_yaml(open('./target/my_model_architecture.yml').read())
model.load_weights('./target/my_model_weights.h5')
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

input_test.test(model)
