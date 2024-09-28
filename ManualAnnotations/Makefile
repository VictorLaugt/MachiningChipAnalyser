IMG_DIR = ../MachiningImages

all: visualize_diagonal3 visualize_vertical

%/manual_measures.csv:
	python image_annotation_utility.py $(IMG_DIR)/$* -m

%/automatic_measures.csv:
	python image_annotation_utility.py $(IMG_DIR)/$* -a -m

quantify_%: %/automatic_measures.csv
	python compare_measures.py $*

clean:
	find . -name '*.bmp' -delete
	find . -name '*.csv' -delete

.PHONY: all clean
