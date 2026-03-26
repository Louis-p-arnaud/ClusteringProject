FROM python:3.11
WORKDIR /app

COPY requierements.txt .
RUN pip install --no-cache-dir -r requierements.txt

COPY . .
EXPOSE 8501

ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=8501
ENV DASHBOARD_PATH_DATA=/app/outputs

CMD ["python", "dashboard.py"]