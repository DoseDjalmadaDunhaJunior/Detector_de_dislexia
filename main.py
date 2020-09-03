# TensorFlow e tf.keras
import tensorflow as tf
from tensorflow import keras

# Librariesauxiliares
import numpy as np
import matplotlib.pyplot as plt

#as duas abaixo são para baixar a base de dados
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#o nome que daremos as nossas categorias
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#aqui é para mostrar uma imagem (muda de acordo com o numero que vc coloca)
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
#plt.show()

# aqui em baixo determina quantas imagens servirão de treinamento e quantas de teste
train_images = train_images / 255.0

test_images = test_images / 255.0


#aqui imprime uma tabela com varias imagens
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
#plt.show()


#aqui começa a rede neural

model = keras.Sequential([
    #a linha abaixo basicamente faz a imagem sair de um quadrado (imagem)
    #para uma linha basicamente
    keras.layers.Flatten(input_shape=(28, 28)),

    #aqui é uma camada de neuronios, com 128 neuronios
    keras.layers.Dense(128, activation='relu'),

    #aqui é outra camada de neuronios, com 10
    keras.layers.Dense(10, activation='softmax')
])

# compilando o modelo
"""
as linhas abaixo são divididas da seguintes formas:
1ª  é como o modelo se atualiza com base no que ve na linha a seguir
2ª é o quão imprecisa é a rede (quanto menor melhor)
3ª monitora os passos de treinamento e teste e o exemplo usa acuracia
"""
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


#treinamento

#o treino é só essa linha abaixo (quanto maior o numero abaixo mais ele treina)
model.fit(train_images, train_labels, epochs=10)

#as 2 linhas abaixo avaliam a acuracia
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

#predições (predição meio que é prever o que vai acontecer)
predictions = model.predict(test_images)

predictions[0]

np.argmax(predictions[0])

test_labels[0]

#a função abaixo é para gerar um grafico da previsão de 10 classes
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')
#ate aqui


#as funções abaixo são para ver a previsão que a rede neural fez ou seja o resultado de seu aprendizado
  i = 0
  plt.figure(figsize=(6, 3))
  plt.subplot(1, 2, 1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(1, 2, 2)
  plot_value_array(i, predictions, test_labels)
  plt.show()

  i = 12
  plt.figure(figsize=(6, 3))
  plt.subplot(1, 2, 1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(1, 2, 2)
  plot_value_array(i, predictions, test_labels)
  plt.show()