pipeline {
  agent {
    docker {
      image 'python:3.8-slim'
    }

  }
  stages {
    stage('Build') {
      steps {
        sh 'pip install --target ${env.WORKSPACE} --user poetry'
        sh 'poetry install'
      }
    }
  }
}