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
        sh 'apt update && apt install -y curl'
        sh 'pip install poetry'
      }
    }
  }
}
