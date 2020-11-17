.PHONY: proyecto.pdf all clean

all: proyecto.pdf

%.tex: %.raw
	./raw2tex $< > $@

%.tex: %.dat
	./dat2tex $< > $@

proyecto.pdf: proyecto.tex
	latexmk -pdf -pdflatex="pdflatex -interaction=nonstopmode" -use-make proyecto.tex

clean:
	latexmk -CA
