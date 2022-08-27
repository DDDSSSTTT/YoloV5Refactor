import tensorflow as tf
import numpy as np
print(tf.__version__)
print(np.__version__)
import tensorflow_hub as hub
hub_model = hub.resolve("../weights/yolov5")
from params_misc import init_params, datasets_from_params
import train

#Create Trainer From Hub Model
keras_model = tf.keras.models.load_model(hub_model)
keras_model.compile(optimizer="Adam", loss="mse", metrics=["mae"])
params = init_params()
trainer = train.Trainer(params)
# reconfg_train_from_keras_model(trainer,keras_model)

#Trainer.Train
train_dataset, valid_dataset = datasets_from_params(params, trainer)
trainer.train(train_dataset, valid_dataset, transfer='scratch', keras_model=keras_model)

#Test