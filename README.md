![Pandas](https://img.shields.io/badge/-Pandas-333333?style=flat&logo=pandas)
![Numpy](https://img.shields.io/badge/-Numpy-333333?style=flat&logo=numpy)
![Matplotlib](https://img.shields.io/badge/-Matplotlib-333333?style=flat&logo=matplotlib)
![Seaborn](https://img.shields.io/badge/-Seaborn-333333?style=flat&logo=seaborn)
![Scikitlearn](https://img.shields.io/badge/-Scikitlearn-333333?style=flat&logo=scikitlearn)

# Modelo de clasificación de hospitalización de pacientes intervenidos por una biopsia prostática

## **Planteamiento de la problemática**

El presente proyecto busca conocer las características más importantes que tienen los pacientes de cierto tipo de enfermedad que terminan en hospitalización. Fue definido como **caso** aquel paciente que fue sometido a biopsia prostática y que en un periodo máximo de 30 días posteriores al procedimiento presentó fiebre, infección urinaria o sepsis; requiriendo manejo médico ambulatorio u hospitalizado para la resolución de la complicación. Por otra parte, se definió como **control** al paciente que fue sometido a biopsia prostática y que no presentó complicaciones infecciosas en el período de 30 días posteriores al procedimiento.

## **Datos disponibles**

Para cumpli con el objetivo de clasificar a los pacientes en si serán hospitalizados o no, se proporcionó una base de datos con 570 registros y 20 variables para analizar. Los datos se ha recopilado teniendo en cuenta: `Antecedentes del paciente`, `Morbilidad asociada al paciente` y `Antecedentes relacionados con la toma de la biopsia`y `Complicaciones infecciosas`. El detalle de cada variable y ortos datos importantes para el entendimiento del problema se encuentra en del [Diccionario de Datos](Diccionario_Datos.mb).

## **Procesamiento de los datos**

Mediante un proceso de ETL se consumieron los datos provistos y se los convitió en un Dataframe donde se realizó una exploración y limpieza inicial, transformación de los datos y carga del conjunto limpio. Luego, mediante un proceso de EDA, se exploró cada una de las variables con el objetivo de entender cada una de ellas y buscar la relación o no con la variable objetivo. A lo largo de la exploración se fueron descartando algunas variables, para finalmente preparar los datos en un formato compatible para la modelación del problema.

En este repositorio se puede encontrar el proceso completo de [ETL](01_ETL.ipynb) y [EDA](02_EDA.ipynb).

## **Modelo de clasificación**

Una vez preparados los datos, se procede a buscar un modelo de Machine Learning de clasificación, que permita clasificar a un paciente como *hospitalizado* o *no hospitalizado*, teniendo en cuenta la información referente a sus antecedentes como paciente, las morbilidad asociadas, los antecedentes relacionados con la toma de la biopsia y complicaciones infecciosas.

Para ello, se plantean dos estrategias para esa busqueda. Por un lado, utilizar las 12 columnas de datos y, en una segunda estrategia, se considera la reducción de dimensionalidad. Para ambas estrategias se utilizan los algoritmos de Árbol de Decisión, K-Vecinos Cernanos y Máquina de Soporte de Vectores.

Para la elección de los mejores hiperparámetros de cada uno de ellos, se utiliza una búsqueda exaustiva mediante la técnica de GridSearch. 

En la notebook [03_modelo](03_modelo.ipynb) se encuentra detallado todo el análisis realizado para llegar al mejor modelo.

## **Conclusiones**

Se logró encontrar un modelo de clasificación que permite la clasificación de la hospitalización o no de un paciente a partir de conocer sus antecedentes, como la edad o relacionados a la intervención en sí de la biopsia, así como morbilidad propia del paciente o cuestiones relacionadas con la aparición de complicaciones infecciosas. El mejor modelo que se obtuvo fue un Árbol de Decisión de máxima profundidad de 19 y utilizando criterio de decisión de Gini, pero previo se tuvo que hacer una reducción de la dimensionalidad utilizando PCA, encontrando 8 Componentes como la mejor opción entre las evaluadas. 

Para este modelo se utilizó la métrica de evaluación F1 Score (Test_F1) dado que combina la precisión (resultados correctos sobre el total de muestras seleccionadas) y la exhaustividad (resultados correctos por sobre los resultados que se buscan identificar) de manera de mantener una relación entre las dos. Se consiguió un F1 Score de 0.92, siendo un muy buen rendimiento considerando el gran desbalance de la clase objetivo.