FROM python:3.11
WORKDIR /app

COPY requierements.txt .
RUN pip install -r requierements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "dashboard_clustering.py", "--server.port=8501", "--server.address=0.0.0.0"]