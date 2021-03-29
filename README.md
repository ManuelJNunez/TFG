<p align="center">
    <img width="639" height="406" src="https://astro.uchicago.edu/depot/images/highlight-080224-3_large.jpg">
</p>

> Imagen obtenida de la web del [Departamento de astronomía y astrofísica de la Universidad de Chicago](https://astro.uchicago.edu/research/auger.php)

# Water Cherenkov Detectors. Cosmic Ray classification.
<p align="center">
    <a href='http://54.161.147.47/job/TFG/job/main/'><img src='http://54.161.147.47/buildStatus/icon?job=TFG%2Fmain'></a>
    <a href="https://www.codacy.com/gh/ManuelJNunez/TFG/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=ManuelJNunez/TFG&amp;utm_campaign=Badge_Coverage"><img src="https://app.codacy.com/project/badge/Coverage/e289951e1da6421e82062829ef76ae5d"/></a>
    <a href="https://www.codacy.com/gh/ManuelJNunez/TFG/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=ManuelJNunez/TFG&amp;utm_campaign=Badge_Grade"><img src="https://app.codacy.com/project/badge/Grade/e289951e1da6421e82062829ef76ae5d"/></a>
    <a href="https://www.gnu.org/licenses/gpl-3.0"><img alt="License: GPL v3" src="https://img.shields.io/badge/License-GPLv3-blue.svg"></a>
    <a href="https://github.com/psf/black"><img alt = "Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

## Descripción del problema

La Tierra está siendo bombardeada constantemente con rayos cósmicos que provienen, como su nombre indica, del universo. Estos rayos se componen de partículas que viajan a la velocidad de la luz, que al entrar en contacto con la atmósfera producen una cascada atmosférica extensa. Dicha cascada emite una luz debido a la radiación gamma llamada "Luz de Cherenkov". A la misma vez, el rayo cósmico se fragmenta formando hadrones. Es decir, estos rayos se descomponen en una componente hadrónica y otra componente electromagnética.

Para poder capturar información sobre dichos rayos cósmicos, se usan WCD (en la imagen de arriba se puede ver uno) que son tanques de agua ultrapura que contienen fotomultiplicadores distribuidos simétricamente dentro del tanque que se utilizan para captar la señal del rayo que incide sobre el mismo. En función de la profundidad del agua del WCD, se emitirá más luz o menos, siendo directamente proporcionales.

El problema que se pretende solucionar en este proyecto, es diseñar un modelo que sea capaz de diferenciar un rayo que sea solo radiación gamma electromagnética de uno que tenga una partícula denominada "muón". Esta partícula se genera por la interacción a alta energía de los hadrones y nos permiten discriminar bastante bien entre la radiación gamma y los hadrones.

## Órdenes del task runner

### Instalación de dependencias

    poetry install

### Ejecutar tests

    poetry run task test

### Ejecutar `pylint`

    poetry run task lint

### Comprobar si el código sigue el estilo `Black`

    poetry run task black

### Visualizar resultado de los tests de cobertura

    poetry run task cov-result

### Generar fichero XML con los resultados del test de cobertura

    poetry run task cov-xml

## Participantes del proyecto
-   **Alumno:** [Manuel Jesús Núñez Ruiz](https://github.com/ManuelJNunez)
-   **Tutor:** [Alberto Guillén Perales](https://github.com/aguillenatc)
