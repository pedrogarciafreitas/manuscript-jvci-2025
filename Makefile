PROJECT = main
PDFLATEX = pdflatex
BIBTEX = bibtex

.PHONY: all
all: build

.PHONY: build
build: $(PROJECT).tex
	@echo "--- Starting LaTeX Compilation for $(PROJECT) ---"
	$(PDFLATEX) $(PROJECT)
	$(PDFLATEX) $(PROJECT)
	$(BIBTEX) $(PROJECT)
	$(PDFLATEX) $(PROJECT)
	$(PDFLATEX) $(PROJECT)
	@echo "--- Compilation Complete. $(PROJECT).pdf is ready. ---"

.PHONY: clean
clean:
	@echo "--- Cleaning intermediate files ---"
	rm -f $(PROJECT).aux $(PROJECT).log $(PROJECT).toc $(PROJECT).out \
	      $(PROJECT).bbl $(PROJECT).blg $(PROJECT).dvi \
	      $(PROJECT).lof $(PROJECT).lot \
	      $(PROJECT).idx $(PROJECT).ilg $(PROJECT).ind \
	      $(PROJECT).gls $(PROJECT).glo $(PROJECT).ist \
	      $(PROJECT).acn $(PROJECT).acr $(PROJECT).alg \
	      $(PROJECT).glg $(PROJECT).run.xml $(PROJECT).synctex.gz \
	      $(PROJECT).glsdefs
	rm -f $(PROJECT).pdfmake 
