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

    stage('linting') {
      steps {
        sh 'poetry run task lint'
      }
    }

    stage('test') {
      steps {
        sh 'poetry run task test'
      }
    }
  }
}