FROM python:3.11-slim

WORKDIR /app

COPY monitoring/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY monitoring/ ./monitoring/

EXPOSE 8502

CMD ["streamlit", "run", "monitoring/app.py", "--server.address", "0.0.0.0", "--server.port", "8502"]