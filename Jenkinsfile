pipeline {
  agent {
    docker {
      image 'python:3.8-slim'
    }
  }

  stages {
    stage('Build') {
      steps {
        sh 'apt-get update && apt-get install -y curl'
        sh 'curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python'
        sh '$HOME/.poetry/bin/poetry install --no-root'
        sh 'poetry install'
      }
    }
  }
}
