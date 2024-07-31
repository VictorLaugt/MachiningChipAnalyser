SRC_DIR = ChipAnalyser
IMG_DIR = imgs
TEST_DIR = ChipAnalyser/tests
TEST_SRC = $(wildcard $(TEST_DIR)/*.py)

export PYTHONPATH += ChipAnalyser

process_%:
	@rm -rf outputs
	@python3 $(SRC_DIR) -i $(IMG_DIR)/$* -o outputs -r

no_render_process_%:
	@rm -rf outputs
	@python3 $(SRC_DIR) -i $(IMG_DIR)/$* -o outputs

test:
	@$(foreach SCRIPT,$(TEST_SRC),\
		printf "\n======= running $(SCRIPT) =======\n";\
		python3 $(SCRIPT);)

test_%:
	@printf "\n======= running $(TEST_DIR)/$*.py =======\n"
	@python3 $(TEST_DIR)/$*.py

clean:
	find . -name __pycache__ -type d | while read -r pycachepath; do rm -rf $$pycachepath; done

.PHONY: test clean
