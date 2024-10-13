SRC_DIR = ChipAnalyser
IMG_DIR = MachiningImages
TEST_DIR = ChipAnalyser/tests
TEST_SRC = $(wildcard $(TEST_DIR)/*.py)

export PYTHONPATH += ChipAnalyser

process_%: clean_outputs
	python $(SRC_DIR) -i $(IMG_DIR)/$* -o outputs_$* -s 3.5 -r

process_without_render_%: clean_outputs
	python $(SRC_DIR) -i $(IMG_DIR)/$* -o outputs_$* -s 3.5

test:
	@$(foreach SCRIPT,$(TEST_SRC),\
		printf "\n======= running $(SCRIPT) =======\n";\
		python $(SCRIPT);)

test_%:
	@printf "\n======= running $(TEST_DIR)/$*.py =======\n"
	@python $(TEST_DIR)/test_$*.py

deployment_test:
	./DeploymentTest/test_deployment.sh

clean_python_cache:
	find . -name __pycache__ -type d | while read -r pycachepath; do rm -rf $$pycachepath; done

clean_outputs:
	rm -rf outputs outputs_* DeploymentTest/outputs DeploymentTest/results.txt

clean: clean_outputs clean_python_cache

.PHONY: test deployment_test clean_outputs clean_python_cache clean
