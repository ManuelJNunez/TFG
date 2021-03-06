pipeline {
  agent {
    dockerfile true
  }

  stages {
    stage('Build') {
      steps {
        checkout scm
        sh 'poetry install'
      }
    }
  }
}
