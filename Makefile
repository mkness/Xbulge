
all: arxiv

arxiv: arxiv.tgz
@PHONY: arxiv

ARXIV := Xbulge.tex \
xbulge-00.pdf xbulge-01.pdf xbulge-fit-data.pdf xbulge-fit-masked.pdf \
xbulge-fit-model.pdf xbulge-fit-resid.pdf xbulge-fit-residmasked.pdf \
xbulge-fit-smooth2.pdf

arxiv.tgz: $(ARXIV)
	tar czf $@ $(ARXIV)

aj:
	mkapj Xbulge


