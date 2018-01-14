joke-dataset-master: jokes/data/download.sh
	bash jokes/data/download.sh data/external

input_filepattern = data/external/joke-dataset-master/*.json
vocab_filename = data/processed/vocab.txt

dataset_prefix = data/processed/jokes
dataset_train = $(dataset_prefix).train
dataset_dev = $(dataset_prefix).dev
dataset_test = $(dataset_prefix).test
model_dir = models/jokes-char-rnn

# TODO(kjchavez): Learn how to properly use a Makefile..

build_vocab: jokes/data/make_dataset.py data/external/joke-dataset-master
	python -m jokes.data.make_dataset build_vocab '$(input_filepattern)' $(vocab_filename)

build_processed: jokes/data/make_dataset.py data/external/joke-dataset-master data/processed/vocab.txt
	python -m jokes.data.make_dataset build_dataset '$(input_filepattern)' $(vocab_filename) $(dataset_prefix)

models/baseline/model.ckpt-1000: jokes/models/main.py data/processed/jokes.train $(vocab_filename)
	python -m jokes.models.main train $(dataset_train) $(vocab_filename) --steps=1000

train: jokes/models/main.py $(dataset_train) $(vocab_filename)
	python -m jokes.models.main train $(dataset_train) $(vocab_filename) --model_dir=$(model_dir)

generate: jokes/models/main.py $(model_dir)/model.ckpt-*
	python -m jokes.models.main generate --model_dir=$(model_dir)

clean_models:
	rm -rf models/*

