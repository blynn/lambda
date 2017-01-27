.PHONY: sync site target
target: site

HS2JS=-mv Main.jsmod /tmp; hastec --opt-all

%.js: %.lhs ; $(HS2JS) $^
%.html: %.lhs ../haskell/menu.html ; cobble mathbook ../haskell/menu $<

LHSNAMES=lambda lisp
LHSFILES=$(addsuffix .lhs, $(LHSNAMES)) $(addsuffix .html, $(LHSNAMES)) $(addsuffix .js, $(LHSNAMES))

site: $(LHSFILES) ../haskell/menu.html

sync: site
	rsync $(LHSFILES) blynn@xenon.stanford.edu:www/haskell/
