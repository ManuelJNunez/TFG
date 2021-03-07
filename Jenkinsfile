pipeline {
  agent {
    docker {
      image 'python:3.8-slim'
    }
  }

  stages {
    stage('Build') {
      steps {
        sh 'pip install poetry'
      }
    }
  }
}
