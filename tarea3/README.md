# CC5114
## Tarea 3: Programación Genética

*  Resumen

En esta tarea se implementan algoritmos para programación genética, que permite desarrollar automáticamente programas que realicen una tarea específica, como buscar un numero, o una expresión en particular. Como los programas se representan a través de árboles, se tuvo que adaptar algunos procesos, como el crossover, mutación y la interpretación de la función de fitness.  Además, se presentan resultados para el heatmap de configuraciones para encontrar un número sin límite de repeticiones. 

Se incluye archivo Tarea_3.py y Tarea_3.ipynb donde se realizaron experimentos.

* Librerías utilizadas
    * random 
    * operator
    * string
    * matplotlib
    * math

* Función de Fitness: 
    Esta función -|x - g| + g, donde x es la evaluación del program y g el valor el objetivo. Cumple que su valor máximo es cuando x = g, lo cual es útil al ahora de interpretar la calidad de un programa. Se decidió ocupar el maximo entre el valor de esta función y el 0 para que no hayan valores muy negativos. 

* Ejercicios 
    * Se usaron como código base los archivos proporcionados por el equipo docente en Ucursos: arboles.py, para la representación de arboles y ast.py para generar programas al azar.

    * Encontrar un número: para este problema se agrego un método a los nodos que retorna una lista con todos los terminales. De tal manera que en la función de fitness si hay algún terminal repetido en la lista, se le castiga asignando puntaje mínimo al individuo. 

    * Sin límite de reproducciones: se considero una función de fitness que no revisa la repetición de terminales, pero que considera el tamaño del arbol como un aspecto negativo de un programa.

    * Sin repeticiones: se procedió de igual manera que en "encontrar un número".

    * Implementación de variables: para la evaluación d eun programa se agrega un diccionario como argumento que define los valores estáticos de las variables. 

    * Symbolic Regression: Se cambia la función de fitness para que pueda recivir una función target como argumento. Se revisan coincidencias del -100 y 100 entre el individuo de prueba y la función objetivo. 

    * Nodo división: se agrega el nodo DivNode que permite la operación. en la función de fitness se castiga a los programas con division por 0 con puntaje mínimo.



* Resultados

    * Se presentan resultados obtenidos para el problema de buscar un número sin límite de repeticiones.
    ![alt text](https://github.com/vicho08/CC5114/blob/master/tarea3/images/encontrar_numero.png "Resultados programa genético")

    * Se presentan resultados obtenidos para el problema de buscar un número sin repeticiones.
    ![alt text](https://github.com/vicho08/CC5114/blob/master/tarea3/images/sin_repeticiones.png "Resultados programa genético")

    * Se presentan resultados obtenidos para el problema symbolic regression.
    ![alt text](https://github.com/vicho08/CC5114/blob/master/tarea3/images/regression.png "Resultados programa genético")

Estos son los resultados obtenidos para heatmap cambiando el tamaño de la población y la tasa de mutación para el problema de encontrar un número sin límite de repeticiones, el resto de los parametros son constantes y se encuentran definidas en el código.
![alt text](https://github.com/vicho08/CC5114/blob/master/tarea3/images/pop_size.png "Heatmap: population size")
![alt text](https://github.com/vicho08/CC5114/blob/master/tarea3/images/mutation_rate.png "Heatmap: mutation rate")

Podemos observar que en el caso del tamaño de la población el mayor decrecimiento en el tiempo se observa entre los 300-400 individios. Se observa que despues de 400 individuos se obtienen resultados rápidamente. 

Para el caso de la tasa de mutación, se observa que entre los valores 0.2 y 0.3 hay decaimiento en el tiempo en el tiempo que necesita. Se observan valores muy variados esto se puede deber a la probabilidad que impide a un árbol seguir cresiendo cuando este se crea. (en el proceso de mutación se realiza un crossover entre el individuo y un árbol generado al azar)
 
```python
genetic_algorithm(250, lambda a, b: fitness_repetion(a,b,ast), ast, 10, mutationRate=i, generations=100, deep= 2, goal=375, heatmap=True)
```





