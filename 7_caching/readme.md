docker run -d --name redis -p 6379:6379 redis:alpine

Start command -
docker start redis

Flush command -
docker exec redis redis-cli FLUSHDB
