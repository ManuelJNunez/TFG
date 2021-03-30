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

    stage('Parallel'){
      parallel {
        stage('Linting') {
          steps {
            sh 'poetry run task lint'
            sh 'poetry run task black'
          }
        }

        stage('Testing') {
          stages{
            stage('Unit Tests') {
              steps {
                sh 'poetry run task test'
              }
            }

            stage('Coverage report') {
              steps {
                sh """
                  poetry run task cov-result
                  poetry run task cov-xml
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
      }
    }
  }
}