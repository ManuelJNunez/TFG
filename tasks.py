from invoke import task
from fabric import Connection
from pathlib import Path
from dotenv import dotenv_values
import sys
import os


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
    train = True

    for child in experiments_path.iterdir():
        if child.suffix in [".yml", ".yaml"]:
            train_files.append(child)

    for file in train_files:
        if not all:
            train = False
            reply = input(
                f"Would you like to run the experiment defined in {str(file)}? [Y/n] "
            )

            if reply.lower() in ["1", "y", "yes", "true", "yeah"]:
                train = True

        if train:
            c.run(f"snapper-ml --config_file={str(file)}", pty=True, env=config)
            c.run("rm -rf artifacts/")


@task
def venvtrain(c):
    c.run("docker-compose up -d optuna-db")
    print()
    c.run("invoke train")
    c.run("docker-compose down")


@task
def dockertrain(c):
    c.run("docker-compose up --build -d", pty=True)
    c.run("docker start -ai tfg_train-container_1")
    c.run("docker-compose down")


@task
def sshtrain(c, destdir=None, host=None, gw=None):
    if host is None or destdir is None:
        sys.exit(
            "Usage: inv sshtrain --destdir=<destination_directory> --host=<server_dir> [--gw=<gw_dir>]"
        )

    cwd = os.getcwd()
    current_dirname = cwd.split("/")[-1]

    gw_connection = Connection(gw) if gw else None
    connect = Connection(host, gateway=gw_connection)

    print("Synchronizing files...")

    c.run(f"rsync -vah $PWD {gw if gw else host}:{destdir}")

    with connect.cd(current_dirname):
        print("\n\nChecking if dependencies are up to date...")
        connect.run("poetry install")
        print()
        connect.run("poetry run inv venvtrain")
