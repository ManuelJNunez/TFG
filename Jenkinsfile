pipeline {
  agent {
    docker {
      image 'python:3.8-slim'
    }
  }

  stages {
    stage('Build') {
      checkout scm
      sh 'python --version'
    }
  }
}
