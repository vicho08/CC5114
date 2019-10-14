# CC5114
## Tarea 1: Entrenar una red neuronal

## Tarea 2: Implementación Algoritmo Genético

*  Resumen

En esta tarea se implementan algoritmos genéticos para resolver el problema Unbound Knapsack y los ejercicios propuestos en clase. Además, se utiliza el algoritmo de la ruleta para seleccionar individuos para que estos puedan dejar herencia usando algoritmos de mutación y crossover clásicos. Se presentan los resultados obtenidos para el problema de la búsqueda de una frase objetivo, indicando el máximo, mínimo y promedio de cada generación. Además, se presentan resultados para el heatmap de configuraciones. 

Se incluye archivo Tarea2- CC5114.py y Tarea2- CC5114.ipynb donde se realizaron experimentos.

* Librerías utilizadas
    * random 
    * math
    * matplotlib
    * pandas: para cargar el dataset.
    * sklearn: para partición dataset en training y test.

* Problema Elegido

Para el problema, se decidió que la representación de un gen es un entero que corresponde a la cantidad de items de tipo i de peso w_i que se intentarán colocar en la mochila de peso máximo C = 15 kg. Este número va de 0 a int(C/w_i) para que se cumpla la restricción del peso. Un individuo es una lista de genes correspondiente a la combinación de items que se intentara colocar en la mochila, como el tipo de item es limitado, tenemos que esta lista es de largo 5. La restricción del peso de la mochila se verifica en cada paso en la creación de un individuo y en la mutación. Pues es preferible que todos los individuos de una población cumplan con la restricción del peso para que tenga sentido aplicar la función ´fitness´. 

A diferencia de los problemas en clase, se prefirio manejar a los individuos como listas y no como strings. Lo que requirió un análisis exhaustivo de tipos en el código. También, se requirió un cambio en la función generador de genes, pues para el problema de la secuencia de bits tenemos un universo posible de genes constante (0-1), también para el problema de strings (abecedario). En cambio, para el problema UK tenemos que el universo depende del tipo del ítem y el peso máximo de la mochila, por lo que el universo de genes depende exclusivamente del problema en cuestión. Este cambio en la función de genes requirió un cambio en el proceso de mutación del problema. 

Para los argumentos del algoritmo genético se agregaron el generador de individuos, tasa de elitismo, la posibilidad de agregar un objetivo conocido para el problema (para el caso de los problemas en clase) y un booleano heatmap para realizar experimentos.

* Resultados

![alt text](https://github.com/vicho08/CC5114/blob/master/tarea2/images/progress.png "Resultados algritmo genético")
![alt text](https://github.com/vicho08/CC5114/blob/master/tarea2/images/pop_size.png "Heatmap: population size")
![alt text](https://github.com/vicho08/CC5114/blob/master/tarea2/images/mutation_rate.png "Heatmap: mutation rate")

Podemos observar que la mayor parte del aprendizaje de la red es entre la segunda y quinta época. (hay una pendiente mayor). Esto también se evidencia en los resultados de la accuracy que va aumentando también entre las mismas épocas. También se observa que la red logró aprender a clasificar los ejemplos del training set. En el código se incluye la predicción realizada para el test set, el cuál logra un 100% de presición. 

```python
red.predecir(X_test, y_test)
error=0.000, acc=100.000
```





