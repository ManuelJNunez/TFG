pipeline {
  agent {
    docker {
      image 'python:3.8-slim'
    }
  }

  stages {
    stage('Build') {
      steps {
        checkout scm
        sh 'pip install poetry'
      }
    }
  }
}
