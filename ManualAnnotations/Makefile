IMG_DIR = ../MachiningImages

all: vertical/automatic_measures.csv diagonal3/automatic_measures.csv quantify_vertical quantify_diagonal3

%/manual_measures.csv:
	python image_annotation_utility.py $(IMG_DIR)/$* -m

%/automatic_measures.csv:
	python image_annotation_utility.py $(IMG_DIR)/$* -a -m

quantify_%: %/manual_measures.csv %/automatic_measures.csv
	python compare_measures.py $*

clean:
	find . -name '*.bmp' -delete
	find . -name '*.csv' -delete

.PHONY: all clean
