pipeline {
  agent {
    docker {
      image 'python:3.8-slim'
      args '--user 0:113'
    }

  }
  stages {
    stage('Build') {
      steps {
        sh 'pip install poetry'
        sh 'python --version'
      }
    }

  }
}