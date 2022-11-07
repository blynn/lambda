.PHONY: sync site target
target: site

HS2JS=-mv Main.jsmod /tmp; hastec -Wall --opt-all

menu.html: menu ; cobble menu menu
%.js: %.lhs ; $(HS2JS) $^
%.html: %.lhs menu.html ; cobble mathbook menu $<

LHSNAMES=index simply hm lisp systemf typo pts sk crazyl pcf natded logski
LHSFILES=$(addsuffix .lhs, $(LHSNAMES)) $(addsuffix .html, $(LHSNAMES)) $(addsuffix .js, $(LHSNAMES))
LHSWNAMES=bohm cl kiselyov matrix
LHSWFILES=$(addsuffix .lhs, $(LHSWNAMES)) $(addsuffix .html, $(LHSWNAMES)) $(addsuffix .wasm, $(LHSWNAMES))

%.o: %.c; $(WCC) $^
%.wasm: %.o; $(WLD) $^ -o $@

HDIR=../boot
bohm.c: bohm.lhs; ($(HDIR)/unlit < $^ ; cat $(HDIR)/inn/BasePrecisely.hs) | $(HDIR)/precisely wasm > $@

cl.c: cl.lhs; ($(HDIR)/unlit < $^ ; cat $(HDIR)/inn/BasePrecisely.hs $(HDIR)/inn/SystemWasm.hs) | $(HDIR)/precisely wasm > $@

kiselyov.c: kiselyov.lhs; ($(HDIR)/unlit < $^ ; cat $(HDIR)/inn/BasePrecisely.hs $(HDIR)/inn/SystemWasm.hs) | $(HDIR)/precisely wasm > $@

matrix.c: matrix.lhs; ($(HDIR)/unlit < $^ ; cat $(HDIR)/inn/BasePrecisely.hs $(HDIR)/inn/SystemWasm.hs) | $(HDIR)/precisely wasm > $@

WCC=clang -O3 -c --target=wasm32 -Wall
WASMLINK=wasm-ld
WLD=$(WASMLINK) --initial-memory=41943040 --export-dynamic --allow-undefined --no-entry

SITE=$(LHSFILES) $(LHSWFILES) menu.html quine.html

site: $(SITE)

sync: $(SITE) ; rsync $^ blynn@xenon.stanford.edu:www/lambda/
