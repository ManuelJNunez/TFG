from invoke import task


@task
def test(c):
    c.run("coverage run --source=src/ml -m pytest")


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

    c.run(command)
