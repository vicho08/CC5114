# CC5114
## Tarea 1: Entrenar una red neuronal

*  Resumen

En esta tarea se implementa una red neuronal simple. La cual es capaz de predecir un valor a partir de un input, y también aprender (ajustar parámetros) a partir del error entre el valor de salida y el esperado. Se ocupo la red con un dataset real: iris, el cual posee tres clases. Se presentan gráficos de error y accuracy durante el entrenamiento de la red. 

* Librerías utilizadas
* * random 
* * math
* * matplotlib
* * pandas: para cargar el dataset.
* * sklearn: para partición dataset en training y test.

* Dificultades encontradas

Fue díficil encontrar errores, pues comprender como funciona la red internamente es complicado. La parte más díficil fue programar la función de entrenamiento para la red, específicamente la parte de backpropagation. Al principio los parametros se actualizaban y tendían a infinito, con lo que la red no funcionaba. El problema era que al actualizar el delta para cada neurona, se estaba sumando el nuevo valor al antiguo y no reasignando el nuevo valor a delta. 

* Resultados

![alt text](https://github.com/vicho08/CC5114/blob/master/tarea1/images/resultados.png "Resultados dataset iris")

Podemos observar que la mayor parte del aprendizaje de la red es entre la segunda y quinta época. (hay una pendiente mayor). Esto también se evidencia en los resultados de la accuracy que va aumentando también entre las mismas épocas. También se observa que la red logró aprender a clasificar los ejemplos del training set. En el código se incluye la predicción realizada para el test set, el cuál logra un 100% de presición. 

```python
red.predecir(X_test, y_test)
error=0.000, acc=100.000
```





