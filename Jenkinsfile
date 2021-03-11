pipeline {
  agent {
    docker {
      image 'mjnunez/testing:latest'
      args '--user jenkins:jenkins'
    }

  }
  stages {
    stage('Build') {
      steps {
        sh 'pip install poetry'
        sh 'poetry install'
      }
    }
  }
}