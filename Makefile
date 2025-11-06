.PHONY: train train-tuned
train:
	python -m src.models.train_mlp --input_csv ar_properties_2022.csv

train-tuned:
	python -m src.models.train_mlp --hidden 128 64 --alpha 0.0005 --lr 0.0007 --max_iter 800
