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

Podemos observar que en el caso del tamaño de la población el mayor decrecimiento en el tiempo se observa entre los 50-200. Sin embargo los mejores resultados se observan alrededor del tamaño 800. esto se debe que mientras mayor sea el tamaño de la poblacion, es más probable que al crear un individuo en la etapa inicial tenga mayores coincidencias con la palabra objetivo.

Para el caso de la tasa de mutación, observamos que más allá de el valor 0.2 el algoritmo no alcanza a encontrar una respuesta antes de las generaciones impuestas. Esto se debe a que se esta permitiendo demasiada diversidad en los individuos con lo que se podría estar perdiendo información ganada sobre la palabra objetivo en generaciones pasadas.

Como reflexión final, es interesate ver como problemas de optimización pueden ser automatizados usando algoritmos genéticos. Pero hay que tener en cuenta que si la función `fitness` o la creación de individuos es demasiado compleja el algoritmo puede ser costosa en tiempo y recursos. Además, para problemas que no se conoce un resultado correcto a priori, como Unbound Knapsack,  para saber si una respuesta es buena se debe comparar con otras respuestas, por lo que no se sabe un criterio para definir un objetivo específico ya que no se conoce la solución. 
```python
genetic_algorithm(pop_size=50, fitness=fitness, generate_gen=generate_char, generate_ind=generate_word, elite=10, mutationRate=0.15, generations=200, goal= "helloworld")
```





