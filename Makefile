# run in python
PYTHON = python3.9

#python 裡面好像原本就有.pyc檔可以做到像.o檔類似的功能

all:
	@echo 1. PM
	@echo 2. test
	@echo 3. NPM
	@echo 4. LSTM
PM:
	$(PYTHON) main.py

test:
	$(PYTHON) test_result.py

NPM:
	$(PYTHON) main_NPM.py

LSTM:
	$(PYTHON) main_LSTM.py

RFT:
	$(PYTHON) main_RFT.py

clean:
	@rm -f $(CURDIR)/tmp_model/*.jpg
	@rm -f $(CURDIR)/tmp_model/*.pth