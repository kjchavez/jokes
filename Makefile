joke-dataset-master: jokes/data/download.sh
	bash jokes/data/download.sh data/external

input_filepattern = data/external/joke-dataset-master/*.json
vocab_filename = data/processed/vocab.txt

build_vocab: jokes/data/make_dataset.py data/external/joke-dataset-master
	python -m jokes.data.make_dataset build_vocab $(input_filepattern) $(vocab_filename)

build_processed: jokes/data/make_dataset.py data/external/joke-dataset-master data/processed/vocab.txt
	python -m jokes.data.make_dataset build_dataset '$(input_filepattern)' $(vocab_filename) data/processed/jokes.dat
