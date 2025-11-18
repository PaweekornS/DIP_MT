## LANTA usage

1. start API service
``` bash
sbatch serving.sh
```

2. connect port with localhost (if you want to test on local)
``` bash
ssh -L 8000:<gpu-node>:8000 <username>@lanta.nstda.or.th
```
Note: you can check gpu-node via myqueue e.g. lanta-g-001

3. test api
``` bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/translate \
  -H "Content-Type: application/json" \
  -d '{
    "wipo_id": 35,
    "english": "Retail services for clothing and footwear.",
  }'
```