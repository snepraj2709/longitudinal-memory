FROM node:22-alpine AS web-build
WORKDIR /build/web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM python:3.13-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    DEMO_BUNDLE=/app/results/demo/demo-v1/bundle.json \
    DEMO_WEB=/app/web/dist
WORKDIR /app
COPY requirements-demo.txt ./
RUN pip install --no-cache-dir -r requirements-demo.txt
COPY src/api/ ./src/api/
COPY results/demo/demo-v1/bundle.json ./results/demo/demo-v1/bundle.json
COPY --from=web-build /build/web/dist ./web/dist
EXPOSE 8000
CMD ["sh", "-c", "python -m uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
