PREPROCESSING_DIR = preprocessing
PREPROCESSING_SRC = $(wildcard $(PREPROCESSING_DIR)/*.py)

export PYTHONPATH += .

debug:
	@echo "PREPROCESSING_DIR = $(PREPROCESSING_DIR)"
	@echo "PREPROCESSING_SRC = $(PREPROCESSING_SRC)"

preprocessing_test:
	@$(foreach SCRIPT,$(PREPROCESSING_SRC),\
		printf "\n======= running $(SCRIPT) =======\n";\
		python3 $(SCRIPT);)

clean:
	rm -f *.avi
	rm -rf __pycache__


.PHONY: debug clean
