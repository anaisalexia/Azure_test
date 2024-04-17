import keras

mnist = keras.datasets.mnist

HEIGHT = 28
WIDTH = 28

(x_train, y_train), (x_test,y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

## MODEL 
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(HEIGHT,WIDTH)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10),
])
# Remarque : Il est possible d'intégrer la fonction tf.nn.softmax dans la fonction d'activation de la dernière couche du réseau. 
# Bien que cela puisse rendre la sortie du modèle plus directement interprétable, cette approche est déconseillée car il est impossible 
# de fournir un calcul de perte exact et numériquement stable pour tous les modèles lors de l'utilisation d'une sortie softmax.
# tf.nn.softmax(predictions).numpy()

#loss
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

#optimizer
optimizer = "adam"

#metrics
metrics = ['accuracy']


model.compile(optimizer=optimizer,
              loss=loss_fn,
              metrics=metrics)

## TRAINING
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test,  y_test, verbose=2)

prediction_model = keras.Sequential([
  model,
  keras.layers.Softmax()
])
# prediction = model(x_train)
print(prediction_model(x_test[:5]))
print(y_test[:5])
