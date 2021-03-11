pipeline {
  agent {
    docker {
      image 'mjnunez/tfgtesting:latest'
    }

  }
  stages {
    stage('Build') {
      steps {
        sh 'pip install poetry'
        sh 'poetry install'
        sh 'python --version'
      }
    }
  }
}