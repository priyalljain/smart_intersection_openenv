FROM python:3.11-slim
WORKDIR /app
COPY server/requirements.txt /app/server/requirements.txt
RUN pip install --no-cache-dir -r /app/server/requirements.txt
COPY . .
EXPOSE 7860
CMD ["openenv", "serve", "--host", "0.0.0.0", "--port", "7860"]