pipeline {
    agent any
    
    environment {
        DOCKER_CREDENTIALS_ID = 'docker-hub-credentials'
        IMAGE_NAME = 'your-docker-username/your-image-name'
    }
    
    stages {
        stage('Checkout') {
            steps {
                git credentialsId: 'github-credentials', url: 'https://github.com/yourusername/your-repo.git'
            }
        }
        
        stage('Code Quality Check') {
            steps {
                sh 'flake8 .'
            }
        }
        
        stage('Unit Testing') {
            steps {
                // Execute unit tests here
            }
        }
        
        stage('Merge to Test') {
            when {
                branch 'dev'
            }
            steps {
                sh 'git checkout test'
                sh 'git merge origin/dev'
                sh 'git push origin test'
            }
        }
        
        stage('Merge to Master') {
            when {
                branch 'test'
            }
            steps {
                script {
                    def merged = sh(script: 'git merge --no-ff origin/test', returnStatus: true)
                    if (merged == 0) {
                        sh 'git push origin master'
                    } else {
                        currentBuild.result = 'FAILURE'
                        error "Merge to master failed"
                    }
                }
            }
        }
        
        stage('Containerize and Push to Docker Hub') {
            when {
                branch 'master'
            }
            steps {
                script {
                    docker.withRegistry('https://registry.hub.docker.com', DOCKER_CREDENTIALS_ID) {
                        def dockerImage = docker.build(IMAGE_NAME)
                        dockerImage.push()
                    }
                }
            }
        }
        
        stage('Email Notification') {
            when {
                branch 'master'
            }
            steps {
                emailext body: 'Jenkins job completed successfully. Image pushed to Docker Hub.', subject: 'CI/CD Pipeline Status', to: 'admin@example.com'
            }
        }
    }
    
    post {
        always {
            cleanWs()
        }
    }
}
