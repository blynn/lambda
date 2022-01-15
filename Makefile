.PHONY: sync site target
target: site

HS2JS=-mv Main.jsmod /tmp; hastec -Wall --opt-all

menu.html: menu ; cobble menu menu
%.js: %.lhs ; $(HS2JS) $^
%.html: %.lhs menu.html ; cobble mathbook menu $<

LHSNAMES=index simply hm lisp systemf typo pts wasm sk crazyl pcf natded logski
LHSFILES=$(addsuffix .lhs, $(LHSNAMES)) $(addsuffix .html, $(LHSNAMES)) $(addsuffix .js, $(LHSNAMES))

HDIR=../boot
bohm.c: bohm.lhs; ($(HDIR)/unlit < $^ ; cat $(HDIR)/inn/BasePrecisely.hs) | $(HDIR)/precisely wasm > $@
bohm.o: bohm.c; $(WCC) $^
bohm.wasm: bohm.o; $(WLD) $^ -o $@

WCC=clang -O3 -c --target=wasm32 -Wall
WASMLINK=wasm-ld-11
WLD=$(WASMLINK) --initial-memory=41943040 --export-dynamic --allow-undefined --no-entry

SITE=$(LHSFILES) menu.html bohm.wasm bohm.lhs bohm.html

site: $(SITE)

sync: $(SITE) ; rsync $^ blynn@xenon.stanford.edu:www/lambda/
