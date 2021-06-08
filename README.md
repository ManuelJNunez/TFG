<p align="center">
    <img width="639" height="406" src="https://astro.uchicago.edu/depot/images/highlight-080224-3_large.jpg">
</p>

> Imagen obtenida de la web del [Departamento de astronomía y astrofísica de la Universidad de Chicago](https://astro.uchicago.edu/research/auger.php)

# Water Cherenkov Detectors. Cosmic Ray classification.
<p align="center">
    <a href='http://jenkins.mjnunez.es/job/TFG/job/main/'><img src='http://jenkins.mjnunez.es/buildStatus/icon?job=TFG%2Fmain'></a>
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

    poetry run invoke test

### Ejecutar `pylint`

    poetry run invoke lint

### Comprobar si el código sigue el estilo `Black`

    poetry run invoke black

### Visualizar resultado de los tests de cobertura

    poetry run invoke cov-result

### Generar fichero XML con los resultados del test de cobertura

    poetry run invoke cov-xml

### Conectarse a través de SSH para entrenar modelos

    poetry run invoke sshtrain --destdir=<destination_directory> --host=<host> [--gw=<gateway>]

## Cómo reproducir el experimento

El experimento ha sido diseñado para que pueda reproducirse entero sólo con unas pocas órdenes. Antes de arrancarlo, hace falta poner en marcha varios servicios, uno de `mlflow` con su correspondiente base de datos `mysql` y otro de `postgresql` para `optuna`.

### Levantar servicio `mlflow`

Para levantar este servicio, se puede hacer de forma local siguiendo [esta guía](https://www.mlflow.org/docs/latest/tracking.html#scenario-1-mlflow-on-localhost) o en una instancia EC2 definida en `terraform/mlflow`. Dicha instancia ejecutará los contenedores con las opciones definidas en el fichero `docker-compose.yml`.

Para desplegarlo en EC2, es necesario un fichero `.env` con las siguientes variables de entorno:

- **MYSQL_RANDOM_ROOT_PASSWORD.** Genera una password aleatoria para el usuario root de la base de datos, por lo que se le debe de asignar cualquier valor (pues será ignorado). También puedes asignarla tú usando la variable `MYSQL_ROOT_PASSWORD` o dejarla vacía usando `MYSQL_ALLOW_EMPTY_PASSWORD` (inseguro).
- **MYSQL_USER.** Nombre de usuario no root para acceder a la base de datos. Ejemplo: *mlflow_user*.
- **MYSQL_DATABASE.** Nombre de la base de datos que contendrá toda la información sobre los experimentos registrados por `mlflow`. Ejemplo: *mlflow_db*.
- **MYSQL_PASSWORD.** Contraseña para la base de datos que contendrá la información de `mlflow`. Ejemplo: *mlflow_pass*.
- **MYSQL_PORT.** Puerto en el que escuchará el servicio `MySQL`, en el `docker-compose.yml` está definido en el *3306*, por lo que si pones uno distinto a ese, actualiza también el otro fichero.
- **MLFLOW_DBHOST.** Debe contener el valor *mlflow-db*, pues así se llama el servicio de `MySQL` en el `docker-compose.yml`.
- **MLFLOW_ARTIFACTS_URI.** URI en el que se guardarán los artefactos, puede ser local o una dirección de S3, como puede verse [aquí](https://www.mlflow.org/docs/latest/tracking.html#scenario-4-mlflow-with-remote-tracking-server-backend-and-artifact-stores). Ejemplos: *file:///home/mjnunez/mlflow* o *s3://your-bucket/path/to/artifacts*.
- **AWS_ACCESS_KEY_ID.** Si usas S3, esto es necesario para poder autentificarte a la hora de consultar los artefactos.
- **AWS_SECRET_ACCESS_KEY.** Si usas S3, esto también es necesario para poder autentificarte, como el anterior.

Tras esto, ejecuta las siguiente órdenes para levantar la instancia de AWS EC2 con `terraform`:

```sh
terraform plan
terraform apply
```

Tras esto, se instalarán todas las dependencias necesarias para ejecutar el servicio (`docker` y `docker-compose`) y copiará los ficheros `.env`, `Dockerfile` y `docker-compose.yml` a la nueva instancia. Dichos archivos también pueden ser desplegados de forma manual en otros dispositivos.

**Nota:** si tras desplegar el servicio con `terraform`, este no inicia en un rato, accede a la instancia y ejecuta la siguiente orden:

```sh
docker-compose up -d
```

### Reproducir el experimento

Tras haber desplegado nuestro servicio de `mlflow`, podemos pasar ahora a reproducir el experimento. Para ello necesitaremos levantar un servicio de `postgresql`, pero gracias a `docker-compose` de nuevo, esto no es un problema.

Antes que nada, para este paso también es necesario crear un `.env`, pero en el root folder del proyecto (dónde se encuentran el `Jenkinsfile` y el `Dockerfile.train`) con las siguientes variables de entorno:

- **POSTGRES_PASSWORD.** Contraseña para acceder a la base de datos `postgres`. Ejemplo: *tu-clave123*.
- **POSTGRES_USER.** Nombre de usuario para acceder a la base de datos. Ejemplo: *user123*.
- **POSTGRES_DB.** Nombre de la base de datos dónde `optuna` guardará sus datos. Ejemplos: *optuna*.
- **MLFLOW_TRACKING_URI.** dirección URL (o IP) de tu servicio `mlflow`. Ejemplos: *http://localhost* o *http://mlflow.tu-dominio.com*.
- **OPTUNA_STORAGE_URI.** URI para acceder a la base de datos `posgres`. Se le puede asignar el valor:
```plaintext  
postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:5432/${POSTGRES_DB}
```
- **AWS_ACCESS_KEY_ID.** Si usas S3 en tu instancia de `mlflow`, para poder autentificarte a la hora de subir los artefactos. También se pueden configurar en `~/.aws/credentials`.
- **AWS_SECRET_ACCESS_KEY.** Igual que la variable anterior, sirve para autentificarte en AWS S3 a la hora de subir artefactos. También se pueden configurar en `~/.aws/credentials`.

Tras haber creado el `.env`, puedes ejecutar el experimento de forma local (con las dependencias instaladas en un entorno virtual gestionado por `poetry`) con las siguientes órdenes:

```sh
poetry install --no-dev
docker-compose up optuna-db -d
poetry run invoke train
```

O simplemente ejecuta lo siguiente, que ya hace todo por ti (menos el install) y al finalizar termina con todos los servicios levantados:
```sh
poetry run invoke venvtrain
```

O si prefieres crearlo en un entorno aún más aislado, puedes ejecutar los experimentos dentro de un contenedor docker si has instalado [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). Para ello, usa la siguiente orden:

```sh
poetry run invoke dockertrain
```

## Participantes del proyecto
-   **Alumno:** [Manuel Jesús Núñez Ruiz](https://github.com/ManuelJNunez)
-   **Tutor:** [Alberto Guillén Perales](https://github.com/aguillenatc)
