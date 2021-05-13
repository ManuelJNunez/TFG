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

    stage('Testing & Linting'){
      parallel {
        stage('Testing') {
          stages{
            stage('Unit Tests') {
              steps {
                sh 'poetry run invoke test'
              }
            }

            stage('Coverage report') {
              steps {
                sh """
                  poetry run invoke covreport
                  poetry run invoke covxml
                """

                withCredentials([string(credentialsId: 'codacy-token', variable: 'CODACY_PROJECT_TOKEN')]) {
                  sh """
                    curl -Ls https://coverage.codacy.com/get.sh > coveragereport.sh
                    chmod 755 coveragereport.sh
                    ./coveragereport.sh report -r coverage.xml
                  """
                }
              }
            }
          }
        }

        stage('Linting') {
          steps {
            sh 'poetry run invoke lint'
            sh 'poetry run invoke black --check'
          }
        }
      }
    }
  }
}