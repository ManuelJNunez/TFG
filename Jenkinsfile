pipeline {
  agent {
    docker {
      image 'python:3.8-slim'
    }
  }

  stages {
    stage('Build') {
      steps {
        sh """
          virtualenv venv --distribute
          . venv/bin/activate
          pip install poetry
        """
        sh 'python --version'
      }
    }
  }
}
