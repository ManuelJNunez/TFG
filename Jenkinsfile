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
      }
    }

    stage('Linting') {
      steps {
        sh 'poetry run task lint'
        sh 'poetry run task black'
      }
    }

    stage('Testing') {
      steps {
        sh 'poetry run task test'
      }
    }
  }
}