PREPROCESSING_DIR = preprocessing
PREPROCESSING_SRC = $(wildcard $(PREPROCESSING_DIR)/*.py)

export PYTHONPATH += .

debug:
	@echo "PREPROCESSING_DIR = $(PREPROCESSING_DIR)"
	@echo "PREPROCESSING_SRC = $(PREPROCESSING_SRC)"

preprocessing_all:
	@$(foreach SCRIPT,$(PREPROCESSING_SRC),\
		printf "\n======= running $(SCRIPT) =======\n";\
		python3 $(SCRIPT);)

preprocessing_%:
	@printf "\n======= running $(PREPROCESSING_DIR)/$*.py =======\n"
	@python3 $(PREPROCESSING_DIR)/$*.py

clean:
	rm -rf results/*
	rm -rf __pycache__


.PHONY: debug clean
