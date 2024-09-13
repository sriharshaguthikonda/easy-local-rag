first create the vault.json file by running the script:

TODO
"c:/Users/deletable/OneDrive/Windows_software/openai whisper/openai/Scripts/python.exe" c:/Users/deletable/OneDrive/easy-local-rag/Vault_json_creation_from_HTMLs.py

TODO

after the vault.json is created
then we need to generate the embeddings

to generate the embeddings run:

"c:/Users/deletable/OneDrive/Windows_software/openai whisper/openai/Scripts/python.exe" c:/Users/deletable/OneDrive/easy-local-rag/Generate_embeddings.pyUsers/deletable/OneDrive/easy-local-rag/Generate_embeddings.py

TODO run the chat app

then we can jus run the chat to chat.

ollama serve

"c:/Users/deletable/OneDrive/Windows_software/openai whisper/openai/Scripts/python.exe" c:/Users/deletable/OneDrive/easy-local-rag/groq_lama_MIlvs_RAG_ETTS.py

streamlit run streamlit_groq_lama_MIlvs_RAG_ETTS.py

TODO: go to the directory
cd C:\Users\deletable\OneDrive\easy-local-rag

TODO Docker start

cd C:\Users\deletable\OneDrive\easy-local-rag
docker compose up -d

TODO Docker stop

docker compose stop

docker ps

TODO attu for milvus

docker run -p 8000:3000 -e MILVUS_URL=172.22.240.1:19530 zilliz/attu:v2.4

grok_api_key
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

forfiles /S /M \* /C "cmd /c if @fsize gtr 52428800 echo @path"
