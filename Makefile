.PHONY: sync site target
target: site

HS2JS=-mv Main.jsmod /tmp; hastec --opt-all

menu.html: menu ; cobble menu menu
%.js: %.lhs ; $(HS2JS) $^
%.html: %.lhs menu.html ; cobble mathbook menu $<

LHSNAMES=index simply hm lisp systemf typo pts wasm
LHSFILES=$(addsuffix .lhs, $(LHSNAMES)) $(addsuffix .html, $(LHSNAMES)) $(addsuffix .js, $(LHSNAMES))

site: $(LHSFILES) menu.html

sync: site
	rsync $(LHSFILES) blynn@xenon.stanford.edu:www/lambda/
