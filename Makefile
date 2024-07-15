TEST_DIR = tests
TEST_SRC = $(wildcard $(TEST_DIR)/*.py)

PREPROCESSING_DIR = preprocessing
PREPROCESSING_SRC = $(wildcard $(PREPROCESSING_DIR)/*.py)

SHAPE_DETECTION_DIR = shape_detection
SHAPE_DETECTION_SRC = $(wildcard $(SHAPE_DETECTION_DIR)/*.py)

export PYTHONPATH += .

all: preprocessing_all shape_detection_all

debug:
	@echo "PREPROCESSING_DIR = $(PREPROCESSING_DIR)"
	@echo "PREPROCESSING_SRC = $(PREPROCESSING_SRC)"

test_all:
	@$(foreach SCRIPT,$(TEST_SRC),\
		printf "\n======= running $(SCRIPT) =======\n";\
		python3 $(SCRIPT);)

test_%:
	@printf "\n======= running $(TEST_DIR)/$*.py =======\n"
	@python3 $(TEST_DIR)/$*.py

preprocessing_all:
	@$(foreach SCRIPT,$(PREPROCESSING_SRC),\
		printf "\n======= running $(SCRIPT) =======\n";\
		python3 $(SCRIPT);)

preprocessing_%:
	@printf "\n======= running $(PREPROCESSING_DIR)/$*.py =======\n"
	@python3 $(PREPROCESSING_DIR)/$*.py

shape_detection_all:
	@$(foreach SCRIPT,$(SHAPE_DETECTION_SRC),\
		printf "\n======= running $(SCRIPT) =======\n";\
		python3 $(SCRIPT);)

shape_detection_%:
	@printf "\n======= running $(SHAPE_DETECTION_DIR)/$*.py =======\n"
	@python3 $(SHAPE_DETECTION_DIR)/$*.py


clean:
	rm -rf results/*
	find . -name __pycache__ -type d | while read -r pycachepath; do rm -rf $$pycachepath; done


.PHONY: debug clean preprocessing_all shape_detection_all
