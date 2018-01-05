joke-dataset-master: jokes/data/download.sh
	bash jokes/data/download.sh data/external

input_filepattern = data/external/joke-dataset-master/*.json
vocab_filename = data/processed/vocab.txt
processed_dataset = data/processed/jokes.dat

build_vocab: jokes/data/make_dataset.py data/external/joke-dataset-master
	python -m jokes.data.make_dataset build_vocab '$(input_filepattern)' $(vocab_filename)

build_processed: jokes/data/make_dataset.py data/external/joke-dataset-master data/processed/vocab.txt
	python -m jokes.data.make_dataset build_dataset '$(input_filepattern)' $(vocab_filename) data/processed/jokes.dat

models/baseline/model.ckpt-1000: jokes/models/main.py data/processed/jokes.dat $(vocab_filename)
	python -m jokes.models.main train $(processed_dataset) $(vocab_filename) --steps=1000

train: jokes/models/main.py data/processed/jokes.dat $(vocab_filename)
	python -m jokes.models.main train $(processed_dataset) $(vocab_filename)

clean_models:
	rm -rf models/baseline

generate: jokes/models/main.py models/baseline/model.ckpt-*
	python -m jokes.models.main generate --model_dir=models/baseline
