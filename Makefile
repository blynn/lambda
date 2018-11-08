.PHONY: sync site target
target: site

HS2JS=-mv Main.jsmod /tmp; hastec -Wall --opt-all

menu.html: menu ; cobble menu menu
%.js: %.lhs ; $(HS2JS) $^
%.html: %.lhs menu.html ; cobble mathbook menu $<

LHSNAMES=index simply hm lisp systemf typo pts wasm sk crazyl pcf natded logski
LHSFILES=$(addsuffix .lhs, $(LHSNAMES)) $(addsuffix .html, $(LHSNAMES)) $(addsuffix .js, $(LHSNAMES))

SITE=$(LHSFILES) menu.html

site: $(SITE)

sync: $(SITE) ; rsync $^ blynn@xenon.stanford.edu:www/lambda/
