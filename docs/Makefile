DOC := proyecto.tex

all: doc

doc: $(DOC:.tex=.pdf)

%.pdf: %.tex
	pdflatex $< && bibtex $* && pdflatex $< && pdflatex $<

clean:
	rm *.aux *.bbl *.blg *.log *.out *.pdf *.toc
