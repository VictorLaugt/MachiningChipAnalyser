TEST_DIR = ChipAnalyser/tests
TEST_SRC = $(wildcard $(TEST_DIR)/*.py)

export PYTHONPATH += ChipAnalyser

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
