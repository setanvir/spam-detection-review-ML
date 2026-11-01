install:
	pip install -r requirements.txt
data:
	python -m src.download --source kaggle
data-uci:
	python -m src.download --source uci
train:
	python -m src.train --data data/emails.csv
train-uci:
	python -m src.train --data data/SMSSpamCollection --out results_uci
test:
	python -m pytest -q
