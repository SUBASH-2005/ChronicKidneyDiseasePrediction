pipeline {

    agent any

    stages {

        stage('Install Dependencies') {
            steps {
                bat 'pip install -r requirements.txt'
            }
        }

        stage('Train Model') {
            steps {
                bat 'python src/train.py'
            }
        }

        stage('Build Docker Image') {
            steps {
                bat 'docker build -t ckd-app .'
            }
        }

        stage('Run Container') {
            steps {
                bat 'docker rm -f ckd-container || exit 0'
                bat 'docker run -d -p 8501:8501 --name ckd-container ckd-app'
            }
        }

    }
}
