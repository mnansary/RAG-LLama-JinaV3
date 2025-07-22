# RAG-LLama-JinaV3

```bash
git clone https://github.com/mnansary/RAG-LLama-JinaV3.git
cd RAG-LLama-JinaV3/ 
conda create -n ragllama python=3.10 -y
conda activate ragllama
pip install -r requirements.txt
```

```bash
cd retriver/
chmod +x run.sh
sudo cp retriever.service /etc/systemd/system/retriever.service
sudo systemctl daemon-reload
sudo systemctl enable retriever.service
sudo systemctl start retriever.service


curl -X 'POST'   'http://localhost:7000/retrieve'   -H 'accept: application/json'   -H 'Content-Type: application/json'   -d '{
  "query": "পাসপোর্ট রিনিউ করতে কি কি লাগে?",
  "k": 2
}'
```