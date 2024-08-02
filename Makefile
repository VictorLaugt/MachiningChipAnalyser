SRC_DIR = ChipAnalyser
IMG_DIR = imgs
TEST_DIR = ChipAnalyser/tests
TEST_SRC = $(wildcard $(TEST_DIR)/*.py)

export PYTHONPATH += ChipAnalyser

process_%: remove_outputs
	python3 $(SRC_DIR) -i $(IMG_DIR)/$* -o outputs -r

no_render_process_%: remove_outputs
	python3 $(SRC_DIR) -i $(IMG_DIR)/$* -o outputs

test:
	@$(foreach SCRIPT,$(TEST_SRC),\
		printf "\n======= running $(SCRIPT) =======\n";\
		python3 $(SCRIPT);)

test_%:
	@printf "\n======= running $(TEST_DIR)/$*.py =======\n"
	@python3 $(TEST_DIR)/test_$*.py

clean: remove_outputs
	find . -name __pycache__ -type d | while read -r pycachepath; do rm -rf $$pycachepath; done

remove_outputs:
	rm -rf outputs outputs_*

.PHONY: test clean remove_outputs
