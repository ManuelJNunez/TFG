from invoke import task, Responder, watchers
from pathlib import Path
from dotenv import dotenv_values


@task
def test(c):
    c.run("coverage run --source=src/ml -m pytest", pty=True)


@task
def covreport(c):
    c.run("coverage report -m")


@task
def covxml(c):
    c.run("coverage xml")


@task
def lint(c):
    c.run("pylint src --generated-members=torch")


@task
def black(c, check=False):
    command = "black src tests"

    if check:
        command += " --check"

    c.run(command, pty=True)


@task
def train(c, all=False):
    experiments_path = Path("src/experiments")
    config = dotenv_values(".env")
    train_files = []

    for child in experiments_path.iterdir():
        if child.suffix in [".yml", ".yaml"]:
            train_files.append(child)

    for file in train_files:
        command = f"snapper-ml --config_file={str(file)}"

        if not all:
            responder = Responder(
                pattern=f"Would you like to run the experiment defined in {str(file)}?",
                response= "y\n",
            )

            c.run(command, pty=True, env=config, watchers=[responder])
        else:
            c.run(command, pty=True, env=config)


@task
def venvtrain(c):
    c.run("docker-compose up optuna-db -d")
    c.run("invoke train")
    c.run("docker-compose down")


@task
def dockertrain(c):
    c.run("docker-compose up --build -d", pty=True)
    c.run("docker start -ai tfg_train-container_1")
    c.run("docker-compose down")
