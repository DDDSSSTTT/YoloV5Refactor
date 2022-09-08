import tensorflow as tf
import tensorflow_hub as hub
from params_misc import init_params, datasets_from_params
import train

#Create Trainer From Hub Model
hub_model = hub.resolve("../weights/test_yolov5")
keras_model = tf.keras.models.load_model(hub_model)
keras_model.compile(optimizer="Adam", loss="mse", metrics=["mae"])

#Init Trainer with params
params = init_params()
trainer = train.Trainer(params)
# reconfg_train_from_keras_model(trainer,keras_model)

#Trainer.Train
train_dataset, valid_dataset = datasets_from_params(params, trainer)
trainer.train(train_dataset, valid_dataset, transfer='scratch', keras_model=keras_model)

#Test