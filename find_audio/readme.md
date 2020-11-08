- Download model from - https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/v0.2/
- Copy the extracted model to ./model directory

python find_audio.py -model model/roberta-base-nli-mean-tokens -model_type sentence_bert -embeddings files/ -query "laughing man."

