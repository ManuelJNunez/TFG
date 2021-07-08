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
  - Administración de configuración
  - Monitorización

---

# DevOps
### CI/CD

- **Continuous Integration (CI).** Automatizar la integración de los cambios del código. Se basa en el uso de herramientas automáticas para verificar que el nuevo código es correcto.
- **Continuous Deployment (CD).** Automatización de la puesta en producción de los nuevos cambios.

---

# Control de versiones


