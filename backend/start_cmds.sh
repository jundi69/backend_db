docker rm influxdb

docker run -d -p 8086:8086 --name influxdb -v influxdb-storage:/var/lib/influxdb2 -e DOCKER_INFLUXDB_INIT_MODE=setup -e DOCKER_INFLUXDB_INIT_USERNAME=admin -e DOCKER_INFLUXDB_INIT_PASSWORD=dingleberries -e DOCKER_INFLUXDB_INIT_ORG=distributed-training -e DOCKER_INFLUXDB_INIT_BUCKET=distributed-training-metrics -e DOCKER_INFLUXDB_INIT_ADMIN_TOKEN=648b65eb0a5b1d7b48e71e695fd6bb6611936548debaf281cf438df8ce03b74b influxdb:2.0

docker rm redis
docker run -d --name redis -p 6379:6379 redis:latest

cd backend
source venv/bin/activate
uvicorn dashboard_api:app --reload --host 0.0.0.0 --port 8000