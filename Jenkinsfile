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
        // stage('Authenticate with Docker Hub') {
        //     steps {
        //         script {
        //             withCredentials([usernamePassword(credentialsId: 'docker-hub-credentials', usernameVariable: 'DOCKER_USERNAME', passwordVariable: 'DOCKER_PASSWORD')]) {
        //                 bat 'docker login -u %DOCKER_USERNAME% -p %DOCKER_PASSWORD%'
        //             }
        //         }
        //     }
        // }
        // stage('Push Docker Image') {
        //     steps {
        //         script {
        //             bat 'docker push gillrafay/mlops_image:latest'
        //         }
        //     }
        // }
    }
}
