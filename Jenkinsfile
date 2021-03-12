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

    stages('test') {
      steps {
        sh 'poetry run task test'
      }
    }
  }
}