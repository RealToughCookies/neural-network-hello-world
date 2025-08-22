.PHONY: train-cnn export predict compare tune clean

train-cnn:
	python -m src.train_fashion_mnist --epochs 2 --batch-size 128 --subset-train 2000 --seed 0 --device cpu --model cnn --outdir artifacts --log-csv artifacts/metrics.csv --plot --confusion-matrix

export:
	python -m src.export --model auto --ckpt artifacts/checkpoints/best.pt --outdir artifacts/export

predict:
	python -m src.predict --backend ts --model-file artifacts/export/model_ts.pt --sample 0

compare:
	python -m src.predict --compare --sample 0

tune:
	python -m src.tune

clean:
	rm -rf artifacts