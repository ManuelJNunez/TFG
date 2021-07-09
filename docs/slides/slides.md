---
marp: true
title: Desarrollo de modelos de Machine Learning aplicando MLOps
description: Aplicando metodologías agiles al Machine Learning
theme: gaia
class:
    - lead
    - invert   
paginate: true
_paginate: false
---

<!-- _class: lead -->

#### Desarrollo de modelos de Machine Learning aplicando MLOps

![bg right 130%](images/mlops.png)

![text](#120)

Manuel Jesús Núñez Ruiz

---

# Índice

1. Objetivos
2. DevOps y MLOps
3. Descripción del problema
4. Herramientas utilizadas
5. Modelos de Deep Learning
6. Infraestructura utilizada
7. Resultados obtenidos
8. Despliegue

<!-- footer: Manuel Jesús Núñez Ruiz -->

---

<!-- _class: lead -->

# 1. Objetivos

---

# Objetivos

- Llevar un control de los experimentos realizados.
- Obtener los mejores resultados posibles mediante el uso de hiperparámetros óptimos.
- Comprobar la validez de un modelo mediante *tests* unitarios (TDD aplicado al ML).
- Despliegue ágil de modelos.
- Permitir la reproducibilidad.

<!-- _footer: Sección 1: Objetivos -->
---

<!-- _class: lead -->

# 2. DevOps y MLOps

---

# DevOps

- *Development* + *Operations*
- Integración entre desarrolladores de sofware y *sysadmins*.
- Software con mayor calidad, menor coste y una altísima frecuencia de *releases*.

<!-- footer: Sección 2: DevOps y MLOps -->
  
---

# DevOps

- Prácticas DevOps:
  - CI/CD
  - Control de versiones
  - Infraestructura como Código
  - Monitorización

---

# DevOps
### CI/CD

- **Continuous Integration (CI).** Automatizar la integración de los cambios del código. Se basa en el uso de herramientas automáticas para verificar que el nuevo código es correcto.
- **Continuous Deployment (CD).** Automatización de la puesta en producción de los nuevos cambios.

---

# DevOps
### Control de versiones

Registra los cambios realizados en los ficheros fuentes a lo largo del tiempo, permitiendo recuperar versiones específicas más adelante.

<br/>

![width:400px](images/gitlogo.png)

---

# DevOps
### Infraestructura como Código

- Permite la gestión y preparación la infraestructura con código.
- Ventajas:
  - Ahorro de tiempo y costes.
  - Facilita la distribución y reproducibilidad.

---

# DevOps

### Monitorización

- Visualizar en tiempo real el rendimiento y estado de las aplicaciones.
- Objetivo: detectar errores lo antes posible.
- Necesario visualizar:
  - Hardware subyacente.
  - Aplicación en ejecución.

---

# MLOps

- Aplicar las prácticas de DevOps al desarrollo de sistemas de ML.
- Problema adicional:
  - Administración de modelos.
  - Administración de hiperparámetros y métricas.
  - Reproducibilidad.

---

<!-- _class: lead -->

# 3. Descripción del problema

<!-- _footer: Manuel Jesús Núñez Ruiz -->

---

<!-- _class: invert -->

# Descripción del problema

- Concepto importante: cascada atmosférica extensa.
- Una partícula muy cargada proveniente del cosmos entra en contacto con la atmósfera.
- Interés: investigar las fuentes de radiación del universo.

![bg right:35% 90%](images/airshower.jpg)

<!-- footer: Sección 3: Descripción del problema -->

---

<!-- _class: invert -->

# Descripción del problema

- Captura de información con Water Cherenkov Detectors.
- Tanques de agua ultrapura con fotomultiplicadores en el fondo.
- Se suelen colocar siguiendo un *layout*.

![bg right:35% 90%](images/wcd.png)

---

<!-- _class: invert -->

# Descripción del problema

- Altura mínima 4.4km.
- Tiempo de funcionamiento esperado: 20 años.
- Coste estimado 40-50 M€.
- Construcción por fases.
- Para evitar derroches se han simulado datos usando CORSIKA.

![bg right:40% 90%](images/layout.png)

---

<!-- _class: invert -->

# Descripción del problema

![bg center:50% 70%](images/iron.png)
![bg center:50% 70%](images/proton.png)


---

# Descripción del problema

<!-- _class: invert -->

- Los datos vienen divididos en dos subconjuntos:
  - Conjunto de entrenamiento (44.971 muestras):
    - Clase 0: 22.481 muestras.
    - Clase 1: 22.490 muestras.
  - Conjunto de pruebas (14.989 muestras):
    - Clase 0: 7.493 muestras.
    - Clase 1: 7.496 muestras.
