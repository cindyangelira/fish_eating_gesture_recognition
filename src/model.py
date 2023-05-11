import joblib
import os
import docker

from utils import FeaturePreprocessing, TimeSeriesClassifier

# Train and save the model
X_train, X_test, y_train, y_test = FeaturePreprocessing().load_data()
model = TimeSeriesClassifier()
model.fit(X_train, y_train)
joblib.dump(model, 'model.pkl')

# Create a Dockerfile for reproducibility
dockerfile = """
FROM python:3.9-slim-buster
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
CMD ["python", "main.py"]
"""

with open('Dockerfile', 'w') as f:
    f.write(dockerfile) 

