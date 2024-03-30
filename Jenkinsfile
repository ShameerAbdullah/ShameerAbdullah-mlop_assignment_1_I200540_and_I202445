pipeline {
    agent any
    stages {
        stage('Build Docker Image') {
            steps {
                script {
                    bat 'docker build -t gillrafay/mlops_image:latest .'
                }
            }
        }
        stage('Authenticate with Docker Hub') {
            steps {
                script {
                    withCredentials([usernamePassword(credentialsId: 'docker-hub-credentials', usernameVariable: 'DOCKERHUB_USER', passwordVariable: 'DOCKERHUB_PASS')]) {
                        bat 'docker login -u %DOCKERHUB_USER% -p %DOCKERHUB_PASS%'
                    }
                }
            }
        }
        stage('Push Docker Image') {
            steps {
                script {
                    bat 'docker push gillrafay/mlops_image:latest'
                }
            }
        }
    }
}
