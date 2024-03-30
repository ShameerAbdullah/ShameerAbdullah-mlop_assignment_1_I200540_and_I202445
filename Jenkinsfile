pipeline {
  agent none
  stages {
    stage('Docker Build') {
      agent any
      steps {
        sh 'docker build -t gillrafay/mlops-assignment1:latest .'
      }
    }
    stage('Docker Push') {
      agent any
      steps {
        withCredentials([usernamePassword(credentialsId: '88c1c648-acca-4ad0-bdae-e47152cfc9af', passwordVariable: 'dockerHubPassword', usernameVariable: 'dockerHubUser')]) {
          sh "docker login -u gillrafay -p ${env.dockerHubPassword}"
          sh 'docker push gillrafay/mlops-assignment1:latest'
        }
      }
    }
  }
}
