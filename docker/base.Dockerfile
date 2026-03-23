FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app/research_agent

COPY research_agent/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY research_agent/ .

CMD ["python", "main.py"]
