# Dynamic Indicators Tools

Conjunto de funciones asociadas al análisis de sistemas dinámicos. Concretamente, enfocado al análisis de estructuras 
invariantes o indicadores dinámicos como los campos FTLE [[1]](#Referencias), Lagrangian Descriptors [[2]](#Referencias) o determinación de 
Lagrangian Structure Coehrents (LCS) [[3]](#Referencias).    

## Cómo comenzar
Las funciones que contiene este paquete están desarroladas utilizando `Python 3.11`.

### Requirements
Los paquetes que utiliza estas funciones son: 
```
scipy>=1.11.4, <1.12.0
matplotlib>=3.7.1,<3.7.2
attrs>=24.2.0,<25.0.0
tqdm>=4.66.0,<4.67.0
pyqt5>=5.15.9
```

### Instalación
```
pip install dynamic-indicators-tools
```

### Indicadores Dinámicos
Los indicadores dinámicos son herramientas matemáticas utilizadas para analizar el comportamiento y la estabilidad de sistemas dinámicos. En este repositorio se incluyen los siguientes indicadores:

- **Finite Time Lyapunov Exponents (FTLE)**: Evalúa la sensibilidad del sistema a condiciones iniciales, ayudando a identificar estructuras coherentes en flujos caóticos.
  -  `ftle_element_wise`: Implementa el cálculo del FTLE en una malla punto a punto.
  -  `ftle_grid`: Implementa el calculo del FTLE directamente en la malla, calculando primero el valor de `f(t, y(t)` para cada punto.
  -  `ftle_variational_equations`: Implementa el cálculo del FTLE directamente en la solución del sistema, integrando el cálculo en el sistema diferencial a traves de las ecuaciones variacionales.
- **Lagrangian Descriptors (LD)**: Evalua la velocidad media del flujo del sistema diferencial para un intervalo de tiempo dado.
  - `lagrangian_descriptors`: Calcula los LD en cada punto integrando el problema (`ld_method=integrate`) o añadiendo el LD dentro del sistema diferencial (`ld_method=differential_equations`).
- **Secciones de Poincaré**: Permiten visualizar trayectorias periódicas y comportamiento a largo plazo en sistemas dinámicos.
  - `poincare_section`: Implementan el cálculo de las secciones de poincaré indicadas en la función `poincare_map`.

### Ejemplos de Implementación
La carpeta [`example`](https://github.com/santhiperbolico/dynamic_indicators_tools/tree/main/examples) contiene ejemplos prácticos de cómo implementar y usar cada uno de los indicadores dinámicos incluidos en este repositorio. 
Estos ejemplos muestran configuraciones aplicables para distintos sistemas dinámicos y ayudan a entender el proceso completo de cálculo y análisis de indicadores como FTLE o Lagrangian Descriptors en situaciones reales.

El ejemplo que podemos encontrar es el del péndulo no lineal amortiguado para cada uno de los indicadores dinámicos disponibles. 

### Contenido
Dado un sistema diferencial, un espacio de fase $\mathcal{D}$ y un tiempo inicial $t_0$, el campo de los *finite-time 
Lyapunov Exponents* (FTLE) o el de los *Lagrangian Descriptors* LD se calcula para cada punto 
$\vec{x}\in U\subset \mathcal{D}$. Bajo esta premisa, necesitamos primero poder resolver el sistema diferencial tomando como
condiciones iniciales cada punto $x\in U$ y de esta manera tener el valor de la función flujo $\phi_{t_0}^{t_0+T}(\vec{x})$.

El código de `dynamic_indicators_tools` se ha estructurado en tres módulos diferenciando las funcionalidades necesarias 
para el cálculo de los indicadores dinámicos.

#### i) differentials\_systems

En este módulo se recogen todos los objetos y funciones destinadas a la resolución numérica de sistemas diferenciales. 
En particular, la construcción de los sistemas se recogen dentro de los objetos `DiffSystem` indicándole la función 
$f(\vec{x}(t), t)$ y la variable del sistema $\vec{x}$ a través de un objeto `DiffVariable`. Los objetos `DiffSystem` 
incluyen un método de resolución del sistema en donde podemos indicarle el integrador que deseemos, donde la solución 
del sistema $\vec{x}($ se almacenará en el objeto `DiffVariable` asociado al sistema. La solución se guarda en el objeto
`DiffVariable` almacenando el valor de la variable $vec{x}(t)$ para una malla de puntos $t\in I\subset [t_0, t_0+T]$. 
Estos objetos contienen un método que, dado un $t^*\notin I$, interpola la solución $\vec{x}(t^*)$ usando los valores 
$vec{x}(t)$ guardado en la malla de valores $I$. 

Lo último destacable de este módulo son los objetos `FlowMap`, los cuales representan las funciones flujo $\phi$ y se 
contruyen utilizando un sistema diferencial `DiffSystem` y el valor que marca el tiempo inicial $t_0$. El objeto 
`FlowMap` es llamable, devolviendo el valor de $\phi_{t_0}^{t_0+T}(\vec{x})$ dado un $\vec{x}\in U$ y un tiempo 
$t=t_0+T$, habiendo definido los parámetros del integrador a utilizar. Además posee un método para, dado unos límites 
de un rectángulo $U\subset \mathcal{D}$, construye una malla de puntos $\phi_{t_0}^{t_0+T}(G)\subset U$  con el valor 
del flujo para cada punto $\vec{x}\in G\subset U$.

Podemos encontrar ejemplos de las funcionalidades que acabamos de describir en los test utnitarios. Dentro de los test 
definidos en `tests/test_diff_systems.py`, se puede observar como se comparan los resultados obtenidos por estos métodos
con las soluciones reales obtenidas del sistema $\dot x_i = (-1)^{i}x_i$. Así, además de tener un ejemplo de como 
utilizar estas funciones, podemos comprobar que la funcionalidad esperada por estos métodos y objetos se cumple con 
exactitud.

#### ii) numercial_methods

En este módulo se han implementado los métodos de integración y de diferenciación numérica. En el último de éstos, 
podemos encontrar en `numercial_methods.differentiation` métodos de derivadas numéricas por diferencias finitas hacia 
delante, con un error de grado uno, y centradas con un error de grado dos. Usando estos métodos se define una función 
que calcula la jacobiana para una malla de puntos dada. Estos métodos se utilizarán para calcular la jacobiana de la 
función flujo y posteriormente el tensor Cauchy-Green.

Las funcionalidades de integración se recogen dentro de `numerical_methods.integrators` los métodos numéricos de 
integración numérica unidimensionales. Estos métodos se basan en la librería `scipy.integrate` y son utilizados en el 
cálculo del campo de *Lagrangian Descriptors*.

Como en el caso anterior, la funcionalidad de los integradores y de las derivadas numéricas son comprobados a través de 
test en `test_lagrangian_descriptors.py` y `test_differentiation.py` respectivamente.

#### iii) dynamic_indicators
En este último módulo se recogen la construcción de los indicadores dinámicos. Aquí podemos destacar dos grupos, 
las construcciones de los FTLE en `finite_time_lyapunov_exponents` y la de los LD `lagrangian_descriptors`. Además se 
recoge las funcionalidades de representación en gráficas dentro de `plot_descriptors` para poder representar los campos 
escalares de cad aindicador.

Por último se define el proceso completo de los indicadores en `dynamic_indicators_process`. Para ello se crea una 
interfaz `DynamicIndicator` que recoge las funcionalidades básicas que debe tener un indicador dinámico, que básicamente
es un método `process`. Posteriormente se construye cada indicador, donde dependiendo de cual sea tendrá su proceso de
cálculo se contruye el método `process`de una manera u otra. Las funcionalidades de este módulo son comporbadas en cada 
uno de los archivos test correspondientes `test_dynamic_indicators_process.py`, `test_finite_time_lyapunov_exponents.py`
y `test_lagrangian_descriptors.py` usando el sistema lineal definido anteriormente.

Haciendo uso de la función `main_process_di` definida en `dynamic_indicators_process`, que dado un archivo json ejecuta 
el método `process` de cada indicador, se puede observar un ejemplo de esta implementación con los sistemas del péndulo 
no lineal simple y amortiguado en la carpeta examples del [repositorio](https://github.com/santhiperbolico/dynamic_indicators_tools).


## Referencias
[1] Shadden, S. C., Lekien, F., & Marsden, J. E. (2005). Definition and properties of Lagrangian coherent structures from finite-time Lyapunov exponents in two-dimensional aperiodic flows. Physica D: Nonlinear Phenomena, 212(3-4), 271-304.

[2] Mancho, A. M., Wiggins, S., Curbelo, J., & Mendoza, C. (2013). Lagrangian descriptors: A method for revealing phase space structures of general time dependent dynamical systems. Communications in Nonlinear Science and Numerical Simulation, 18(12), 3530-3557.

[3] Haller, G. (2011). A variational theory of hyperbolic Lagrangian coherent structures. Physica D: Nonlinear Phenomena, 240(7), 574-598.

## Autor

  - **Santiago Arran Sanz**
    ([santhiperbolico](https://github.com/santhiperbolico/))
