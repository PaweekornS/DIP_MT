## LANTA usage
- This API is running on port 8000

1. start API service
``` bash
sbatch serving.sh
```

2. connect port with localhost (if you want to test on local)
``` bash
ssh -L <localport>:<gpu-node>:8000 <username>@lanta.nstda.or.th
```
Note: you can check gpu-node via `myqueue` e.g. lanta-g-001

3. test api
``` bash
curl http://localhost:<localport>/health

curl -X POST http://localhost:<localport>/translate \
  -H "Content-Type: application/json" \
  -d '{
    "wipo_id": 35,
    "english": "Retail services for clothing and footwear.",
  }'
```

## Local usage
1. run docker
```bash
docker-compose build
docker-compose up
```
2. test api
- health check
``` bash
curl http://localhost:8000/health
```
- translate En2Th
```bash
curl -X POST http://localhost:8000/translate_en2th \
  -H "Content-Type: application/json" \
  -d '{
    "wipo_id": 35,
    "english": "Retail services for clothing and footwear.",
  }'
```
- translate Th2En
```bash
curl -X POST http://localhost:8000/translate_th2en \
  -H "Content-Type: application/json" \
  -d '{
    "wipo_id": 35,
    "thai": "สินค้าและบริการค้าปลีก",
  }'
```
