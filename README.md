# Toxic Comment Moderation MLOps System

## Project Overview

This project is an end-to-end MLOps system for identifying toxic comments. It uses the Jigsaw Toxic Comment Classification dataset and classifies comments into six categories:

- toxic
- severe_toxic
- obscene
- threat
- insult
- identity_hate

The goal of the project was not just to train a machine learning model, but to build out the full lifecycle around it. The project includes experiment tracking and model versioning, a prediction API, persistent storage, a user-facing application, automated testing and CI, Docker containers, AWS deployment, and a separate monitoring dashboard.

The final system is deployed on AWS, with separate EC2 instances running the FastAPI backend, the user-facing Streamlit app, and the Streamlit monitoring dashboard.

## System Architecture

The project has three main components that work together:

1. **User Frontend** — A Streamlit app where users can enter a comment and see how the model classifies it across the six toxicity categories. After getting a prediction, users can also provide feedback on whether they think the prediction was correct.

2. **ML Backend** — A FastAPI application that handles the actual predictions. It loads the Production model from the Weights & Biases Model Registry, sends the comment through the model, returns the results to the frontend, and logs the prediction data and user feedback in Amazon DynamoDB.

3. **Monitoring Dashboard** — A separate Streamlit app that uses the prediction and feedback data stored in DynamoDB to keep track of how the model is performing. The dashboard shows total predictions, feedback responses, live accuracy, prediction latency over time, and the distribution of predicted toxicity classes.

The basic flow of the application is:

User → Streamlit Frontend → FastAPI Backend → W&B Production Model  
                                     ↓  
                             Amazon DynamoDB  
                                     ↓  
                     Streamlit Monitoring Dashboard

Each part of the project runs in its own Docker container and is deployed on a separate AWS EC2 instance.

## Dataset & Model Development

The model was trained using the Jigsaw Toxic Comment Classification dataset. The training data contains roughly 160,000 comments, with each comment labeled across six possible toxicity categories: toxic, severe_toxic, obscene, threat, insult, and identity_hate.

Because a single comment can belong to more than one category, this is a multi-label classification problem. For example, a comment could be classified as both toxic and insulting at the same time.

For the baseline model, I used TF-IDF to turn the comment text into numerical features and trained a multi-output logistic regression model to make predictions for each of the six toxicity categories.

I evaluated the model using exact match accuracy along with micro and macro F1 scores. The baseline model produced:

- **Exact Match Accuracy:** 0.8776
- **Micro F1:** 0.6724
- **Macro F1:** 0.5402

I also experimented with the logistic regression hyperparameters and found that using `C=2.0` improved the overall results. The tuned model produced:

- **Exact Match Accuracy:** 0.8808
- **Micro F1:** 0.6847
- **Macro F1:** 0.5574

Based on these results, the tuned model was selected as the model to move forward with for the production system.

## Experiment Tracking & Model Registry

I used Weights & Biases (W&B) to keep track of the model experiments and manage different versions of the trained model. For each training run, I logged the model configuration and performance metrics so I could compare the results between experiments.

The tracked information included the model hyperparameters, exact match accuracy, micro and macro F1 scores, the Git commit associated with the run, and a version of the training data. This made it easier to see not only which model performed better, but also changes between training runs.

The trained models were also saved as W&B artifacts and managed through the W&B Model Registry. The baseline model was registered first, followed by the improved model using `C=2.0`.

After comparing the results, I promoted the tuned model to **Production** in the registry. When it starts, the FastAPI application loads the version tagged as Production. This means FastAPI uses the Production model version directly from W&B.

## FastAPI Backend & DynamoDB

I used FastAPI to build the backend that receives comments from the Streamlit frontend, sends them to the Production model for classification, and returns the prediction results back to the user.

The main endpoints are:

- **`/health`** — Provides a simple way to check that the API is running.
- **`/predict`** — Accepts a user comment, sends it through the Production model, and returns a yes/no prediction for each of the six toxicity categories.
- **`/feedback`** — Records whether the user thought the model's prediction was correct.

Amazon DynamoDB is used as the project's persistent data store. Each prediction is logged in DynamoDB along with information such as the request ID, comment text, prediction results, timestamp, and prediction latency.

When a user answers the "Was this prediction correct?" question in the Streamlit app, the feedback is also saved with the corresponding prediction in DynamoDB. This gives the monitoring dashboard both prediction data and real user feedback to work with.

The FastAPI application runs on its own AWS EC2 instance and uses an IAM role to access DynamoDB without storing AWS credentials directly on the server.

## User Frontend & Monitoring Dashboard

### User Frontend

I built the user-facing part of the project with Streamlit. A user can enter a comment and submit it to the FastAPI backend, then see a yes/no prediction for each of the six toxicity categories.

After seeing the results, the user is asked, "Was this prediction correct?" and can select Yes or No. Only one feedback response can be submitted for each prediction. The response is sent back through the FastAPI backend and saved in DynamoDB, where it can be used to monitor how the model is performing with real user feedback.

### Monitoring Dashboard

I built a second Streamlit application specifically for monitoring the model in production. This application runs separately from the user frontend and connects directly to DynamoDB to use the prediction and feedback data collected by the system.

The monitoring dashboard displays:

- **Total Predictions** — The total number of predictions logged in DynamoDB.
- **Feedback Responses** — The number of predictions that have received user feedback.
- **Live Accuracy** — The percentage of feedback responses where the user indicated that the prediction was correct.
- **Prediction Latency Over Time** — Shows how long prediction requests are taking.
- **Prediction Class Distribution** — Shows how often the different toxicity categories are being predicted.

The frontend and monitoring dashboard run on separate AWS EC2 instances. They do not exchange monitoring data through local files; DynamoDB provides the persistent data that connects the prediction and monitoring parts of the system.

## Testing & CI/CD

To make sure the main parts of the application continue to work as changes are made, I added automated testing. I used pytest to create both unit and integration tests for the project.

The current tests check:

- **Text preprocessing** — Confirms that the text-cleaning function handles input as expected.
- **Health endpoint** — Confirms that the FastAPI `/health` endpoint responds successfully.
- **Prediction endpoint** — Tests that `/predict` returns the expected response structure and prediction results.
- **Feedback endpoint** — Tests that user feedback can be submitted successfully.

For the API tests, I used mock versions of the model and database functions so the tests can run without connecting to W&B or AWS, while also preventing test data from being added to the production DynamoDB table.

I used Ruff for code linting and GitHub Actions for continuous integration. The CI workflow automatically runs Ruff and the full pytest test suite when a pull request is opened against the `main` branch.

This means each proposed code change gets automatically checked before it is merged into `main`. If the linting or tests fail, the CI check will fail and the problem can be fixed before the change is added to the main branch.

## Docker & AWS Deployment

I used Docker to package the different parts of the project so they could run consistently both locally and on AWS. The project includes a FastAPI backend and two Streamlit applications—one for the user frontend and one for the monitoring dashboard. Each component has its own Dockerfile and requirements.

Before deploying to AWS, I built and tested each Docker image locally to make sure the applications worked correctly inside their containers.

For the AWS deployment, I used three separate EC2 instances:

- **API EC2 Instance** — Runs the Dockerized FastAPI backend on port 8000. The instance uses an IAM role to access DynamoDB, while the W&B API key is provided to the container as an environment variable.
- **Frontend EC2 Instance** — Runs the Dockerized Streamlit user application on port 8501. This is where users enter comments, view the model's predictions, and submit feedback. The Streamlit app sends prediction and feedback requests to the FastAPI backend.
- **Monitoring EC2 Instance** — Runs the Dockerized Streamlit monitoring dashboard on port 8502. The dashboard reads prediction and feedback data directly from DynamoDB and displays the production monitoring metrics.

On each EC2 instance, I pulled the project files from GitHub and used them to build and run the Docker container for that part of the application. This keeps each part of the application separate while still allowing the three pieces to work together as one system.

## Local Setup

The project can also be run locally for development and testing. Python, Git, and Docker should be installed before getting started.

### 1. Clone the Repository

Clone the GitHub repository and move into the project folder:

```bash
git clone https://github.com/crystaline10/toxic-comments-mlops.git
cd toxic-comments-mlops
```

### 2. Create a Virtual Environment

Create and activate a Python virtual environment before installing the packages needed. 
```bash
python -m venv .venv
.venv\Scripts\activate
```
Then install the dependencies for the FastAPI backend, Streamlit frontend, and monitoring dashboard: 
```bash
pip install -r api/requirements.txt
pip install -r frontend/requirements.txt
pip install -r monitoring/requirements.txt
```

### 3. Configure Required Credentials

The FastAPI backend needs access to W&B to load the Production model and to AWS DynamoDB to save prediction and feedback data. 
The Streamlit monitoring dashboard also needs access to AWS so it can read the prediction and feedback data stored in DynamoDB.

To connect to these services, the W&B API key is stored as an environment variable, while AWS credentials provide access to DynamoDB. These credentials are kept separate from the project code and should never be committed to GitHub.

### 4. Run the FastAPI Backend

From the root of the project, start the FastAPI backend:

```bash
uvicorn api.main:app --reload
```

The API will be available at:

http://127.0.0.1:8000

The interactive FastAPI documentation can be viewed at:

http://127.0.0.1:8000/docs

### 5. Run the Streamlit Frontend

In a second terminal, start the Streamlit frontend:

```bash
streamlit run frontend/app.py
```

The Streamlit user application will be available at:

http://localhost:8501

### 6. Run the Monitoring Dashboard

In yet another terminal, activate the virtual environment, and start the Streamlit monitoring dashboard:
```bash
streamlit run monitoring/app.py --server.port 8502
```

The monitoring dashboard will display the prediction and feedback data stored on DynamoDB.
The dashboard will be available at:

http://localhost:8502

## Example User Requests

There are two ways a user can interact with the model:

- **Streamlit Frontend** — Enter a comment through the Streamlit app and get back the model's predictions.
- **FastAPI API** — Send a comment directly to the `/predict` endpoint using FastAPI's Swagger interface.

### Using the Streamlit Frontend

In the Streamlit app, a user enters a comment into the text box and selects **Analyze Comment**. For example:

**Example comment:**  
`Your dog is a complete idiot and I hate it.`

The model analyzes the comment and returns a yes/no prediction for each of the six toxicity categories, with green or red dots used to make the results easy to read. The user is then asked, "Was this prediction correct?" and can provide feedback by selecting Yes or No. The prediction and feedback data are stored in DynamoDB and used by the Streamlit monitoring dashboard to track how the model is performing.

### Using the FastAPI API

You can also get a prediction directly through the `/predict` endpoint. FastAPI's Swagger interface at `http://127.0.0.1:8000/docs` provides an easy way to test the endpoint.

An example request body is:

```json
{
  "text": "You are a complete idiot and I hate you."
}
```

The API returns the prediction results for all six toxicity categories, along with a request ID and prediction latency. The request ID connects the prediction to any feedback the user provides, while the latency is used to monitor how long the model takes to respond. 


## AWS Deployment Steps

To run the project on AWS, I used three separate EC2 instances: one for the FastAPI backend, one for the Streamlit user frontend, and one for the Streamlit monitoring dashboard.

### 1. Create the EC2 Instances

Create three EC2 instances for the project:

- **API Instance** — Runs the FastAPI backend on port `8000`.
- **Frontend Instance** — Runs the Streamlit user application on port `8501`.
- **Monitoring Instance** — Runs the Streamlit monitoring dashboard on port `8502`.

Configure the EC2 security groups to allow the traffic needed by each application. SSH access is also needed to connect to the instances and set up the project.

### 2. Connect to Each EC2 Instance

Connect to each instance using SSH and the private key associated with the EC2 instances.

Example:

```bash
ssh -i "your-key.pem" ec2-user@your-ec2-public-dns
```

### 3. Install Git and Docker

Install Git and Docker on each EC2 instance. Start Docker and configure it to start automatically with the instance.

```bash
sudo dnf install git -y
sudo dnf install docker -y
sudo systemctl start docker
sudo systemctl enable docker
```

### 4. Clone the GitHub Repository

Clone the project repository onto each EC2 instance:

```bash
git clone https://github.com/crystaline10/toxic-comments-mlops.git
cd toxic-comments-mlops
```

Each EC2 instance uses the same project repository but builds and runs the Docker container needed for its part of the application.

### 5. Deploy the FastAPI Backend

On the API instance, build the FastAPI Docker image:

```bash
sudo docker build -f api/Dockerfile -t toxic-comments-api .
```

Provide the W&B API key as an environment variable and run the API container:

```bash
sudo docker run -d --name toxic-comments-api \
  --restart unless-stopped \
  -p 8000:8000 \
  -e WANDB_API_KEY="$WANDB_API_KEY" \
  toxic-comments-api
```

The API EC2 instance connects to DynamoDB through an IAM role, so AWS credentials do not need to be stored in the project code.

### 6. Deploy the Streamlit Frontend

On the frontend instance, build the frontend Docker image:

```bash
sudo docker build -f frontend/Dockerfile -t toxic-comments-frontend .
```

Run the container and provide the address of the FastAPI instance:

```bash
sudo docker run -d --name toxic-comments-frontend \
  --restart unless-stopped \
  -p 8501:8501 \
  -e API_URL=http://<API-EC2-PUBLIC-IP>:8000 \
  toxic-comments-frontend
```

The Streamlit frontend can then send prediction and feedback requests to the FastAPI backend.

### 7. Deploy the Streamlit Monitoring Dashboard

On the instance used for monitoring, build the monitoring Docker image:

```bash
sudo docker build -f monitoring/Dockerfile -t toxic-comments-monitoring .
```

Run the monitoring container:

```bash
sudo docker run -d --name toxic-comments-monitoring \
  --restart unless-stopped \
  -p 8502:8502 \
  toxic-comments-monitoring
```

I used an IAM role to give the monitoring EC2 instance access to the prediction and feedback data stored in DynamoDB.

### 8. Verify the Deployment

Check that each Docker container is running:

```bash
sudo docker ps
```

Once all three containers are running, the full system works together to accept user comments, return toxicity predictions through FastAPI, collect user feedback, store the prediction and feedback data in DynamoDB, and display the results in the separate Streamlit monitoring dashboard.

## Repository Structure

The project is organized into separate folders for each part of the MLOps workflow:

```text
toxic-comments-mlops/
├── .github/
│   └── workflows/        # GitHub Actions CI workflow
├── api/                  # FastAPI backend and API Docker setup
├── frontend/             # Streamlit user application
├── monitoring/           # Streamlit monitoring dashboard
├── tests/                # Unit and integration tests
├── training/             # Model training code
├── .dockerignore
├── .gitignore
├── pyproject.toml        # Ruff configuration
└── README.md
```

The api/, frontend/, and monitoring/ folders each contain the files needed to build and run that part of the application. The training/ folder contains the model training code, while the tests/ folder contains the automated tests used by the CI workflow.

## Project Links

- **GitHub Repository:** https://github.com/crystaline10/toxic-comments-mlops
- **Weights & Biases Project:** https://wandb.ai/crystal-mcnama-university-of-denver/toxic-comments-api

## Summary

This project brought together the different parts of the MLOps lifecycle into one working system. Starting with model training and experiment tracking, I built out a FastAPI backend, DynamoDB storage, Streamlit applications for users and monitoring, automated testing and CI, Docker containers, and an AWS deployment.

In the finished system, a user can submit a comment, see the toxicity predictions from the Production model in W&B, and provide feedback on the results. The prediction and feedback data are then saved in DynamoDB and used to keep track of how the model is performing over time.