# Tarea 1

La tarea 1 consiste en crear una red neuronal que se alimente de datasets de clasificación y devuelva gráficos de [error, precision] vs epoch y confusion matrix

## Correr

Para correrlo, use 

```bash
python run Tarea1.py
```

y correrá automáticamente, imprimiendo 20 gráficos en pantalla debido a:
- 2 datasets (Iris y Breast Cancer)
- 5 particiones KFold-Cross Validation
- 2 tipos de gráficos ([error, precision] vs epoch y confusion matrix)

Alternativamente, puede usar Run en Pycharm.

Se usaron las librerías:

- numpy para operar datos
- sklearn.metrics para la confusion_matrix
- sklearn.datasets para los datasets load_breast_cancer y load_iris
- sklearn.model_selection para KFold
- matplotlib.pyplot para los gráficos

## Análisis

La red esta implementada con tres clases principales: NeuralNetwork, NeuronLayer y Perceptron, representando cada uno de los elementos de una red neuronal.

Respecto a las dificultades que hubo al hacer la tarea, trabajar con las distintas dimensiones de los elementos y comprender cuando obtener los datos de error y precisión fue motivo de consultas a un auxiliar.

La red es relativamente rápida para datasets pequeños, demorándose en correr la tarea completa alrededor de 5 minutos, con 200 epochs el trabajo sobre Iris, y 100 epochs el trabajo sobre Breast Cancer.

Para que las redes que trabajan con el dataset de iris dieran resultados más acertados, se tuvo que incrementar la cantidad de epochs de 100 a 200, pues entregaba casi siempre resultados erróneos.

En los gráficos de error y precisión, el eje X representa los epoch sobre los que se entrenó la red, mientras que el eje y representa el valor de precisión (naranjo) o error (azul).

En la matriz de confusión, se muestran los labels trabajados y el reconocimiento que tuvo real.

La red funciona mejor a más epochs se le de para entrenar, para ello al final del archivo Tarea1.py, en las líneas 261 y 266 puede modificar la cantidad de epochs dado.

Iris Confusion Matrix: 
![alt text](https://raw.githubusercontent.com/jorgelobos/cc5114/feature/Tarea1/iris_confusion.png "Iris Confusion Matrix")

Iris Error: 
![alt text](https://raw.githubusercontent.com/jorgelobos/cc5114/feature/Tarea1/iris_error.png "Breast Cancer error")

Breast Cancer Confusion Matrix: 
![alt text](https://raw.githubusercontent.com/jorgelobos/cc5114/feature/Tarea1/breast_confusion.png "Breast Cancer Confusion Matrix")

Breast Cancer Error: 
![alt text](https://raw.githubusercontent.com/jorgelobos/cc5114/feature/Tarea1/breast_error.png "Breast Cancer Error")
