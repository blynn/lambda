== Crazy L ==

Our next compiler generates WebAssembly for a family of languages related to
https://tromp.github.io/cl/lazy-k.html[Lazy K], which in turn is a union of
minimalist languages based on combinator calculus.

[pass]
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
<script src="crazyl.js"></script>
<script type="text/javascript">
function runWasmInts(a) {
  WebAssembly.instantiate(new Uint8Array(a), {i:
    {f:x => Haste.putChar(x),
     g:Haste.getChar,
     h:x => Haste.putInt(x)}}).then(x => x.instance.exports.e());
}
</script>
<button id="natB">Factorial</button>
<button id="lazykB">Reverse</button>
<button id="fussykB">Const Q</button>
<button id="crazylB">Length</button>
<br>
<textarea hidden>
  # Sort a list in Crazy L
  # Too slow :(
Y=ssk(s(k(ss(s(ssk))))k)
z=\n.n(\x.sk)k
V=\xyf.fxy
A=\p.pk
B=\p.p(sk)
N=sk
c=\htcn.ch(tcn)
P=\nfx.n(\gh.h(gf))(\u.x)(\u.u)
L=\mn.(\pq.pqp)(z(nPm))((\pab.pba)(z(mPn)))
H=\l.l(\ht.h)N
u=\l.l(\ht.k)(sk)
f=\xp.u(Ap)(L(H(Ap))x(V(Ap)(cx(Bp)))(VN(cx(c(H(Ap))(Bp)))))(VN(cx(Bp)))
r=\xl.(\q.(u(Aq)(c(H(Aq))(Bq))(Bq)))(lf(V(cxN)N))
\l.lrN
</textarea>
<textarea id="source" rows="10" cols="80">
</textarea>
<textarea id="natDemo" hidden>
t=\xy.x
f=\xy.y
l=\fx.fx
p=\nfx.n(\gh.h(gf))(\u.x)(\u.u)
m=\mnf.m(nf)
z=\n.n(\x.f)t
Y=\f.(\x.xx)(\x.f(xx))
a=Y(\fn.(zn)l(mn(f(pn))))
a(\fx.f(f(f(fx))))  # Compute 4 factorial.
</textarea>
<textarea id="lazykDemo" hidden>
1111100011111111100000111111111000001111111000111100111111000111111100011110011111000111111100011110011100111111100011110011111111100011111111100000111111111000001111111100011111111100000111111111000001111111000111111100011110011111000111001111111110000011111110001111001111110001111111110000011100111001111111000111100111111000111100111111000111111100011111110001111111110000011110011100111100111111100011110011111100011111111100000111001111111000111100111111000111100111111000111100111001111111000111111100011110011111000111111100011110011111100011110011111000111111100011111110001111001111100011111110001111001110011111110001111001111100011111110001111001110011111110001111111110000011111111100000111100111111100011110011111100011111110001111001111100011111110001111001111110001111111110000011111111000111110001111110001111111110000011110011100111111100011110011100111001111001111001111111000111111111000001111001111001111111110000011110011111111000111111111000001111111110000011111111000111111111000001111111110000011111110001111111000111100111110001110011111111100000
</textarea>
<textarea id="fussykDemo" hidden>
Q=(\fx.f(f(f(fx))))(\fx.f(f(fx)))  # 3^4 = 81 = ASCII 'Q'.
e=s(skk)(skk)(s(skk)(skk)(s(s(ks)k)(skk)))   # 256.
v=s(k(s(k(s(k(s(k(ss(kk)))k))s))(s(skk))))k  # Pairing.
k(vQ(vee))  # Ignore input, and output "Q".
</textarea>
<textarea id="crazylDemo" hidden>
u=\nfx.f(nfx)
m=\mnf.m(nf)
z=m(\fx.f(f(fx)))((\fx.f(f(f(fx))))(\fx.f(fx)))
c=\htcn.ch(tcn)
\l.c(l(ku)z)i
</textarea>
<br>
<input type="radio" id="natRadio"    name="mode">Nat<br>
<input type="radio" id="lazykRadio"  name="mode">Lazy K<br>
<input type="radio" id="fussykRadio" name="mode">Fussy K<br>
<input type="radio" id="crazylRadio" name="mode" checked>Crazy L<br>
<button id="compB">Compile</button>
<br>
<br>
<b>intermediate form</b>:
<br>
<textarea id="sk" rows="5" cols="80" readonly></textarea>
<br>
<b>wasm</b>:
<br>
<textarea id="asm" rows="5" cols="80" readonly></textarea>
<br>
<button id="runB">Run</button>
<p>
Input: <textarea id="input" rows="1" cols="16"></textarea>
Output: <textarea id="output" rows="1" cols="16" readonly></textarea>
</p>
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

In these languages, all variables must be one character, and instead of
defining `main`, we expect an expression at the top-level. Comments begin with
`#` and all whitespace except newlines are ignored.

== Vanishingly Small ==

The syntax of link:sk.html[SKI combinator calculus] is already terse, but we
can pare it down further.

For starters, we can use
https://en.wikipedia.org/wiki/Polish_notation[Polish notation] to replace pairs
of parentheses with a single symbol. The
https://en.wikipedia.org/wiki/Unlambda[Unlambda] language chooses the
backquote, while https://esolangs.org/wiki/Iota[Iota] chooses the
asterisk. (Thus obtaining languages that are even more serious about prefix
notation than Lisp.)

We squeeze the syntax harder by playing with combinators.

In Iota, we define the combinator $\iota = \lambda x . x S K$.
It's handy to define $V = \lambda x y z . z x y$, which Smullyan calls the
Vireo, or the pairing bird, so we can write $\iota = V S K$.
Then

\[
\iota \iota = \iota S K = S S K K = S K (K K) = I
\]

from which we deduce $\iota (\iota \iota) = S K$,
$\iota (\iota (\iota \iota)) = K$, and
$\iota (\iota (\iota (\iota \iota))) = S$.

https://esolangs.org/wiki/Jot[The Jot language] goes even further, managing to
handle the S combinator, K combinator, and arbitrary application order with
just two symbols:

\[
\begin{align}
[]   &= I \\
[F0] &= \iota [F] = [F]S K \\
[F1] &= \lambda x y.[F](x y) = S(K[F])
\end{align}
\]

where $[F]$ represents the decoding of a string $F$ of 0s and 1s. In
particular, the empty string is a valid program: it represents $I$, the
identity combinator.

Later, our `dumpJot` function will show how to encode SK programs in Jot.

Incidentally, https://en.wikipedia.org/wiki/Iota_and_Jot[the description of Jot
on Wikipedia seems erroneous] (as of May 2017). I get the impression that
$\iota = \lambda w.w S K$ is confused with $\lambda w.S(K w)$,
so while $w0$ indeed denotes $\iota w$, $w1$ actually denotes $S(K w)$.

Also, in general, $0^* w$ differs from $I w$, which is only a minor issue for
Gödel numbering: like floating point numbers, we can tacitly assume the
presence of a leading 1 bit. All the same, there must be some reason for
decoding from the end of the string instead of the beginning, and it would be
nice if leading zeroes could be omitted without changing a program's meaning.

== Input and Output ==

For lambda calculus, the most natural choice for representing a byte is to
Church-encode it as a natural number between 0 and 255.

One choice for representing a string of bytes is to use nested Church-encoded
pairs. https://tromp.github.io/cl/lazy-k.html[Lazy K] takes this approach.

What about the end of the list? Lazy K demands lazy evaluation, which allows
it to require the input list be infinite in length, with all values after the
input string be (the Church encoding of) 256. For example, the string "AB"
would be represented as:

------------------------------------------------------------------------------
V (church 65) (V (church 66) (V (church 256) (V (church 256) (...))))
------------------------------------------------------------------------------

where, as before, `Vxyz = zxy`, and `church n` is the Church encoding of the
number `n`. From now on, we drop the `church`, and assume numbers are
Church-encoded.

== Fussy K ==

Unfortunately, the reference implementation of Lazy K is sloppy with respect
to the output. Ideally, it should look for `V 256 x` in the
output list for any value of `x`, at which point the program should terminate,
but instead, the current item of the list is tested by applying it to the K
combinator, and if this returns 256 then the program halts. Indeed, the
documentation explicitly mentions that `K 256` is a valid end-of-output
marker.

However, in general `V 256 x` behaves differently to `K 256`, because
`V 256 x (KI) = x` while `K 256 (KI) = 256`. This turns out to complicate our
implementation below.

We tie up this loose end by defining Fussy K to be the Lazy K language as it
is specified, that is, the output list must be terminated with a 256 in the
first argument of a V combinator: `K 256` will not do.

== Crazy L ==

Lazy K combines the syntaxes of SKI combinator calculus, Unlambda, Iota, and
Jot, which amazingly coexist mostly in peace. The only exception is the `i`
program, which Lazy K interprets as the identity combinator rather than the
iota combinator.

To this, we add lambda abstractions and top-level definitions, where all
variables must be single characters.

We also change the encoding of the input and output strings. Pairs are
overkill for representing lists of bytes. After all, we can build arbitrary
binary trees with pairs. Instead, we use
https://en.wikipedia.org/wiki/Church_encoding#Represent_the_list_using_right_fold[the right fold representation of lists].
List manipulations become elegant. A minor additional benefit is that lazy
evaluation is no longer mandatory.

We name the resulting language Crazy L.

== Parsing ==

We define the V combinator, successor combinator, and right fold combinator
in terms of the S and K combinators. Our code would probably be faster if we
made them part of our language instead, but let's start simple.

\begin{code}
{-# LANGUAGE CPP #-}
#ifdef __HASTE__
{-# LANGUAGE OverloadedStrings #-}
import Control.Concurrent.MVar
import Haste.DOM
import Haste.Events
import Haste.Foreign
import Data.Bool
#else
{-# LANGUAGE TemplateHaskell #-}
import System.Console.Readline
import System.Environment
import System.IO
import Test.QuickCheck.All
#endif
import Control.Monad
import Data.Char
import Data.List
import Data.Maybe
import Text.ParserCombinators.Parsec
import Text.Read

infixl 5 :@
data Term = Var String | Term :@ Term | Lam String Term

instance Show Term where
  show (Var s)  = s
  show (l :@ r)  = show l ++ showR r where
    showR t@(_ :@ _) = "(" ++ show t ++ ")"
    showR t          = show t

vireo = mustParse "s(k(s(k(s(k(s(k(ss(kk)))k))s))(s(skk))))k"
foldBird = mustParse "s(k(s(k(s(ks)(s(k(s(ks)k)))))(s(skk))))k"
skk = Var "s" :@ Var "k" :@ Var "k"
vsk = vireo :@ Var "s" :@ Var "k"
succBird = Var "s" :@ (Var "s" :@ (Var "k" :@ Var "s") :@ Var "k")
\end{code}

Our expression parser closely follows
https://tromp.github.io/cl/lazy-k.html[the grammar specified in the description
of Lazy K]. The only differences are that we support lambda abstractions and
treat unreserved letters as variables.

\begin{code}
ccexpr :: Parser Term
ccexpr = do
  xs <- many expr
  pure $ case xs of
    [] -> skk
    _  -> foldl1 (:@) xs

expr     = const skk <$> char 'i' <|> expr'
iotaexpr = const vsk <$> char 'i' <|> expr'
expr' = jotRev . reverse <$> many1 (oneOf "01")
    <|> const skk <$> char 'I'
    <|> Var . pure . toLower <$> oneOf "KS"
    <|> Var . pure <$> letter
    <|> between (char '(') (char ')') ccexpr
    <|> (char '`' >> (:@) <$> expr <*> expr)
    <|> (char '*' >> (:@) <$> iotaexpr <*> iotaexpr)
    <|> flip (foldr Lam) <$> between (char '\\' <|> char '\0955') (char '.')
      (many1 $ (:[]) <$> var) <*> ccexpr

var = lookAhead (noneOf "skiSKI") >> letter

jotRev []       = skk
jotRev ('0':js) = jotRev js :@ Var "s" :@ Var "k"
jotRev ('1':js) = Var "s" :@ (Var "k" :@ jotRev js)

data Top = Super String Term | Main Term

top :: Parser Top
top = do
  t <- try super <|> Main <$> ccexpr
  eof
  pure t

super = do
  name <- pure <$> var
  char '='
  Super name <$> ccexpr

parseLine = parse top "" . filter (not . isSpace) . takeWhile (/= '#')

mustParse s = t where Right (Main t) = parseLine s

parseProgram s = case mEnv of
  Left err -> Left $ "parse error: " ++ show err
  Right env -> case lookup "main" env of
    Nothing -> Left "missing main"
    Just m  -> Right $ sub env m
  where
    mEnv = map f <$> mapM parseLine (lines s)
    f (Super s rhs) = (s,        babs rhs)
    f (Main term)   = ("main", babs term)
mustParseProgram s = t where Right t = parseProgram s
\end{code}

== Bracket Abstraction ==

Again, we use https://tromp.github.io/cl/LC.pdf[the bracket abstraction rules
described by Tromp] to transform the source to an intermediate form consisting
of S and K combinators only.

\begin{code}
fv vs (Var s) | s `elem` vs = []
              | otherwise   = [s]
fv vs (x :@ y)              = fv vs x `union` fv vs y
fv vs (Lam s f)             = fv (s:vs) f

babs (Lam x e)
  | Var "s" :@ Var"k" :@ _ <- t = Var "s" :@ Var "k"
  | x `notElem` fv [] t         = Var "k" :@ t
  | Var y <- t, x == y          = Var "s" :@  Var "k" :@ Var "k"
  | m :@ Var y <- t, x == y, x `notElem` fv [] m = m
  | Var y :@ m :@ Var z <- t, x == y, x == z =
    babs $ Lam x $ Var "s" :@ Var "s" :@ Var "k" :@ Var x :@ m
  | m :@ (n :@ l) <- t, isComb m, isComb n =
    babs $ Lam x $ Var "s" :@ Lam x m :@ n :@ l
  | (m :@ n) :@ l <- t, isComb m, isComb l =
    babs $ Lam x $ Var "s" :@ m :@ Lam x l :@ n
  | (m :@ l) :@ (n :@ l') <- t, l `noLamEq` l', isComb m, isComb n =
    babs $ Lam x $ Var "s" :@ m :@ n :@ l
  | m :@ n <- t = Var "s" :@ babs (Lam x m) :@ babs (Lam x n)
  where t = babs e
babs (Var s) = Var s
babs (m :@ n) = babs m :@ babs n

sub env (x :@ y) = sub env x :@ sub env y
sub env (Var s)  | s `elem` ["s", "k"]    = Var s
                 | Just t <- lookup s env = sub env t
                 | otherwise              = error $ "no binding for " ++ s

isComb e = null $ fv [] e \\ ["s", "k"]

noLamEq (Var x) (Var y) = x == y
noLamEq (a :@ b) (c :@ d) = a `noLamEq` c && b `noLamEq` d
noLamEq _ _ = False
\end{code}

== Interpreter ==

For the command-line version, we interpret the program instead of compiling it.

A term returns either a Church-encoded number or a string, encoded with
Church pairs or the right fold.
One solution is to introduce a corresponding data type:

\begin{code}
#ifndef __HASTE__
data RunValue = I Int | S String
\end{code}

For all languages except Lazy K, we can introduce special combinators
to do most of the heavy lifting:

  * For Nat, let `0` represent the number 0, and let `(+)` be a combinator
  that assumes its argument is a number and increments it. Then given a
  Nat program `p`, we recover its output by reducing `p(+)0`, since its
  output is Church-encoded.

  * For Fussy K, we maintain a string that is initially empty. Let `(!)` be a
  combinator such that  `(!)xy` runs `x(+)0`. If the result is 256, then we
  output the string and terminate, otherwise we append the corresponding byte
  to a buffer, and run `y(!)`. Then given a Fussy K program `p` and its input
  `q`, we interpret it by reducing `pq(!)`.

  * For Crazy L, again we maintain a string that is initially empty. Define
  the combinator `(>)` so that `(>)xy` appends the byte corresponding to
  `x(+)0` to the string then reduces `y`. Let `(.)` be a combinator that
  returns the string when run. Then we interpret a Crazy L program `p` run on
  an input `q` by reducing `pq(>)(.)`.

\begin{code}
run (m :@ n) stack = run m (n:stack)
run (Var "k") (x:_:stack)   = run x stack
run (Var "s") (x:y:z:stack) = run x $ z:(y :@ z):stack
run (Var "+") [x] | I n <- run x [] = I $ 1 + n
run (Var "0") [] = I 0
run (Var "!") [x, y] | I n <- run x [Var "+", Var "0"] = S $ case n of
  256 -> []
  _   -> chr n : t where S t = run y [Var "!"]
run (Var ">") [x, y] = S $ chr n : t where
  I n = run x [Var "+", Var "0"]
  S t = run y []
run (Var ".") [] = S []
run e s = error $ show e
\end{code}

An alternative is to just use String. We can encode a number as a
single-character string. However, this limits the Nat language to programs
whose outputs are less than 256:

\begin{code}
run' (m :@ n) stack = run' m (n:stack)
run' (Var "k") (x:_:stack)   = run' x stack
run' (Var "s") (x:y:z:stack) = run' x $ z:(y :@ z):stack
run' (Var "+") [x] | ord c == 255 = []
                   | otherwise = [succ c]
                   where [c] = run' x []
run' (Var "0") [] = [chr 0]
run' (Var "!") [x, y] | [c] <- run' x [Var "+", Var "0"] = c:run' y [Var "!"]
                      | otherwise = []
run' (Var ">") [x, y] = run' x [Var "+", Var "0"] ++ run' y []
run' (Var ".") [] = []
run' e s = error $ show e
\end{code}

We provide functions to convert the input into Church numerals, one-pair
lists, and right-fold lists:

\begin{code}
church 0 = Var "k" :@ skk
church n = Var "s" :@ (Var "s" :@ (Var "k" :@ Var "s") :@ Var "k")
  :@ church (n - 1)

pList []     = vireo :@ church 256     :@ pList []
pList (x:xs) = vireo :@ church (ord x) :@ pList xs

rfList s = foldr (\a b -> foldBird :@ a :@ b)
  (Var "s" :@ Var "k") $ church . ord <$> s
\end{code}

The Lazy K interpreter encodes the input as a Church one-pair list, and feeds
this to the given term `u` along with the K combinator. We use the `(+)` and
`0` combinators to recover the integer represented by a Church numeral. If this
is 256 then execution is halted, otherwise we record the corresponding byte and
recurse on `uSK`.

\begin{code}
lazyK t inp = g (t :@ pList inp) where
  g u = case run u [Var "k", Var "+", Var "0"] of
    I 256 -> []
    I n   -> chr n : g (u :@ (Var "s" :@ Var "k"))
\end{code}

The other languages are thin wrappers around combinators we have already
defined.

We also add a ``Nat to Nat'' language for programs that expect a Church numeral
and return a Church numeral.

\begin{code}
fussyK t inp = s where S s = run t [pList inp, Var "!"]
crazyL t inp = s where S s = run t [rfList inp, Var ">", Var "."]
succ0 t _ = show n where I n = run t [Var "+", Var "0"]
nat2nat t inp = show n where
  I n = run t [church $ fromMaybe 0 $ readMaybe inp, Var "+", Var "0"]
\end{code}

== Testing ==

During development, it was useful to see combinators expressed in different
forms:

\begin{code}
dumpSK t _ = show t
dumpIota t _ = f t where
  f (x :@ y)  = '*':f x ++ f y
  f (Var "k") = "*i*i*ii"
  f (Var "s") = "*i*i*i*ii"
dumpJot t _ = f t where
  f (x :@ y)  = '1':f x ++ f y
  f (Var "k") = "11100"
  f (Var "s") = "11111000"
dumpUnlambda t _ = f t where
  f (x :@ y)  = '`':f x ++ f y
  f (Var "k") = "k"
  f (Var "s") = "s"
\end{code}

We test our code with QuickCheck on
https://tromp.github.io/cl/lazy-k.html[known Lazy K examples]:

\begin{code}
rev = concat [
  "1111100011111111100000111111111000001111111000111100111111000111111",
  "1000111100111110001111111000111100111001111111000111100111111111000",
  "1111111110000011111111100000111111110001111111110000011111111100000",
  "1111111000111111100011110011111000111001111111110000011111110001111",
  "0011111100011111111100000111001110011111110001111001111110001111001",
  "1111100011111110001111111000111111111000001111001110011110011111110",
  "0011110011111100011111111100000111001111111000111100111111000111100",
  "1111110001111001110011111110001111111000111100111110001111111000111",
  "1001111110001111001111100011111110001111111000111100111110001111111",
  "0001111001110011111110001111001111100011111110001111001110011111110",
  "0011111111100000111111111000001111001111111000111100111111000111111",
  "1000111100111110001111111000111100111111000111111111000001111111100",
  "0111110001111110001111111110000011110011100111111100011110011100111",
  "0011110011110011111110001111111110000011110011110011111111100000111",
  "1001111111100011111111100000111111111000001111111100011111111100000",
  "1111111110000011111110001111111000111100111110001110011111111100000"]

pri = concat [
  "K",
  "(SII(S(K(S(S(K(SII(S(S(KS)(S(K(S(KS)))(S(K(S(S(KS)(SS(S(S(KS)K))(KK)))))",
  "(S(S(KS)(S(KK)(S(KS)(S(S(KS)(S(KK)(S(KS)(S(S(KS)(S(KK)(SII)))",
  "(K(SI(KK)))))))(K(S(K(S(S(KS)(S(K(SI))(S(KK)(S(K(S(S(KS)K)(S(S(KS)K)I)",
  "(S(SII)I(S(S(KS)K)I)(S(S(KS)K)))))(SI(K(KI)))))))))(S(KK)K)))))))(K(S(KK)",
  "(S(SI(K(S(S(S(S(SSK(SI(K(KI))))(K(S(S(KS)K)I(S(S(KS)K)(S(S(KS)K)I))",
  "(S(K(S(SI(K(KI)))))K)(KK))))(KK))(S(S(KS)(S(K(SI))(S(KK)(S(K(S(S(KS)K)))",
  "(SI(KK))))))(K(K(KI)))))(S(S(KS)(S(K(SI))(SS(SI)(KK))))(S(KK)",
  "(S(K(S(S(KS)K)))(SI(K(KI)))))))))(K(K(KI))))))))))(K(KI)))))(SI(KK)))))",
  "(S(K(S(K(S(K(S(SI(K(S(K(S(S(KS)K)I))(S(SII)I(S(S(KS)K)I)))))))K))))",
  "(S(S(KS)(S(KK)(SII)))(K(SI(K(KI)))))))(SII(S(K(S(S(KS)(S(K(S(S(SI(KK))",
  "(KI))))(SS(S(S(KS)(S(KK)(S(KS)(S(K(SI))K)))))(KK))))))(S(S(KS)",
  "(S(K(S(KS)))(S(K(S(KK)))(S(S(KS)(S(KK)(SII)))(K(S(S(KS)K)))))))(K(S(S(KS)",
  "(S(K(S(S(SI(KK))(KI))))(S(KK)(S(K(SII(S(K(S(S(KS)(S(K(S(K(S(S(KS)(S(KK)",
  "(S(KS)(S(K(SI))K))))(KK)))))(S(S(KS)(S(KK)(S(K(SI(KK)))(SI(KK)))))",
  "(K(SI(KK))))))))(S(S(KS)(S(K(S(KS)))(S(K(S(KK)))(S(S(KS)(S(KK)(SII)))",
  "(K(SI(K(KI))))))))(K(K(SI(K(KI)))))))))(S(K(SII))(S(K(S(K(SI(K(KI))))))",
  "(S(S(KS)(S(KK)(SI(K(S(K(S(SI(K(KI)))))K)))))(K(S(K(S(SI(KK))))",
  "(S(KK)(SII)))))))))))(K(SI(K(KI))))))))(S(S(KS)K)I)",
  "(SII(S(K(S(K(S(SI(K(KI)))))K))(SII)))))"]

kk256 = "k(k(s(skk)(skk)(s(skk)(skk)(s(s(ks)k)(skk)))))"

prop_rev s = lazyK (mustParse rev)   t == reverse t where t = take 10 s
prop_id s  = lazyK (mustParse "")    s == s
prop_emp s = lazyK (mustParse kk256) s == ""
prop_pri   = "2 3 5 7 11 13" `isPrefixOf` lazyK  (mustParse pri) ""
prop_pri'  = "2 3 5 7 11 13" `isPrefixOf` fussyK (mustParse pri) ""

fac = unlines [
  "Y=ssk(s(k(ss(s(ssk))))k)",
  "P=\\nfx.n(\\gh.h(gf))(\\u.x)(\\u.u)",
  "M=\\mnf.m(nf)",
  "z=\\n.n(\\x.sk)k",
  "Y(\\fn.zn(\\fx.fx)(Mn(f(Pn))))"]

prop_fac   = nat2nat (mustParseProgram fac) "5" == "120"

return []
runAllTests = $quickCheckAll
\end{code}

== Command-line UI ==

A REPL glues the above together. The first two command-line arguments determine
the language (or dump format) and the input to the program; if omitted, they
default to Lazy K and ``Hello, World!''.  Lines of the program itself are read
from standard input using GNU Readline.

\begin{code}
repl lang inp env = do
  let rec = repl lang inp
  ms <- readline "> "
  case ms of
    Nothing -> putStrLn ""
    Just s  -> do
      addHistory s
      case parseLine s of
        Left err  -> do
          putStrLn $ "parse: " ++ show err
          rec env
        Right (Super s rhs) -> do
          let t = babs rhs
          putStrLn $ s ++ "=" ++ show t
          rec ((s, t):env)
        Right (Main term) -> do
          putStrLn $ lang (sub env $ babs term) inp
          rec env

main = do
  as <- getArgs
  let
    f lang = g lang $ case tail as of
      []    -> ""
      (a:_) -> a
    g lang inp = hSetBuffering stdout NoBuffering >> repl lang inp []
  if null as then g lazyK "Hello, World!" else case head as of
    "n"     -> f succ0
    "n2n"   -> f nat2nat
    "lazy"  -> f lazyK
    "fussy" -> f fussyK
    "crazy" -> f crazyL
    "sk"    -> f dumpSK
    "iota"  -> f dumpIota
    "jot"   -> f dumpJot
    "unl"   -> f dumpUnlambda
    "test"  -> void runAllTests
    bad     -> putStrLn $ "bad command: " ++ bad
\end{code}

== Compiler ==

For the webpage edition, we compile the intermediate form of the code into
WebAssembly.

We adopt link:sk.html[our previous strategy]. We encode an SK expression as
a binary tree, which we store in linear memory and reduce. Each node consists
of two 4-byte numbers. these numbers represent the left and right children
respectively. Certain negative values represent combinators, while all
other values are pointers.

In order to handle the program input, we precompute the SK representations of
the Church numerals from 0 to 256, along with the successor, pairing, and right
fold combinators.  These live at the beginning of linear memory.

\begin{code}
#else
encodeTree e = gen ++ toArr (length gen) e
addrSucc = 257 * 8
codeSucc = toArr addrSucc succBird
addrVireo = addrSucc + length codeSucc
codeVireo = toArr addrVireo vireo
addrRFold = addrVireo + length codeVireo
codeRFold = toArr addrRFold foldBird

genChurch = enCom "s" ++ enCom "k" ++ concat [toU32 addrSucc ++ toU32 (n * 8) | n <- [0..255]]

gen = genChurch ++ codeSucc ++ codeVireo ++ codeRFold
\end{code}

Below we pick special values for each combinator, and describe how to
encode an SK expression as a tree in linear memory. The `(<)` combinator is
for streaming input: when reduced, it ignores its first argument and returns
the next byte of input attached to another `(<)` combinator. In Lazy K and
Fussy K, we use a Church pair to join them. In Crazy L, we use a right fold.

\begin{code}
enCom "0" = neg32 1
enCom "+" = neg32 2
enCom "k" = neg32 3
enCom "s" = neg32 4
enCom ">" = neg32 5
enCom "." = neg32 6
enCom "<" = neg32 8
toArr n (Var a :@ Var b) = enCom a ++ enCom b
toArr n (Var a :@ y)     = enCom a ++ toU32 (n + 8) ++ toArr (n + 8) y
toArr n (x     :@ Var b) = toU32 (n + 8) ++ enCom b ++ toArr (n + 8) x
toArr n (x     :@ y)     = toU32 (n + 8) ++ toU32 nl ++ l ++ toArr nl y
  where l  = toArr (n + 8) x
        nl = n + 8 + length l
neg32 n = [256 - n, 255, 255, 255]
toU32 = take 4 . byteMe
byteMe n | n < 256   = n : repeat 0
         | otherwise = n `mod` 256 : byteMe (n `div` 256)
\end{code}

== Machine Code ==

Again, we define constants and utility functions to help with the wasm binary
format:

\begin{code}
br       = 0xc
br_if    = 0xd
getlocal = 0x20
setlocal = 0x21
teelocal = 0x22
i32load  = 0x28
i32store = 0x36
i32const = 0x41
i32eq    = 0x46
i32ne    = 0x47
i32lt_u  = 0x49
i32ge_u  = 0x4f
i32add   = 0x6a
i32sub   = 0x6b
i32mul   = 0x6c
i32shl   = 0x74
i32shr_s = 0x75
i32shr_u = 0x76

leb128 n | n < 64   = [n]
         | n < 128  = [128 + n, 0]
         | otherwise = 128 + (n `mod` 128) : leb128 (n `div` 128)

varlen xs = leb128 $ length xs

lenc xs = varlen xs ++ xs

sect t xs = t : lenc (varlen xs ++ concat xs)

encStr s = lenc $ ord <$> s

encType "i32" = 0x7f
encType "f64" = 0x7c

encSig ins outs = 0x60  -- Function type.
  : lenc (encType <$> ins) ++ lenc (encType <$> outs)
\end{code}

We have a few more imports this time. Loosely speaking:

  * `f` returns a character.
  * `g` asks for the next character.
  * `h` returns an integer.

\begin{code}
nPages = 8
compile mode e = concat [
  [0, 0x61, 0x73, 0x6d, 1, 0, 0, 0],  -- Magic string, version.
  -- Type section.
  sect 1 [encSig ["i32"] [], encSig [] [], encSig [] ["i32"]],
  -- Import section.
  sect 2 [
    -- [0, 0] = external_kind Function, type index 0.
    encStr "i" ++ encStr "f" ++ [0, 0],
    -- [0, 2] = external_kind Function, type index 2.
    encStr "i" ++ encStr "g" ++ [0, 2],
    encStr "i" ++ encStr "h" ++ [0, 0]],
  -- Function section.
  -- [1] = Type index.
  sect 3 [[1]],
  -- Memory section.
  -- 0 = no-maximum
  sect 5 [[0, nPages]],
  -- Export section.
  -- [0, 1] = external_kind Function, function index 3.
  sect 7 [encStr "e" ++ [0, 3]],
\end{code}

The assembly is only a little more elaborate than our previous version.
We initialize the instruction pointer to the beginning of the program,
which is placed just after the precomputed trees.

\begin{code}
  -- Code section.
  let
    ip = 0  -- program counter
    sp = 1  -- stack pointer
    hp = 2  -- heap pointer
    ax = 3  -- accumulator
    bx = 4
    ccount = 6  -- Number of non-default cases in main br_table.
  in sect 10 [lenc $ [1, 5, encType "i32",
    i32const] ++ varlen gen ++ [setlocal, ip,
    i32const] ++ leb128 (65536 * nPages) ++ [setlocal, sp,
    i32const] ++ varlen heap ++ [setlocal, hp] ++ case mode of
\end{code}

The cost of Lazy K's sloppiness is more apparent in assembly. We need an
extra outer loop to test the input with K. Recall the other languages can
be implemented with special-purpose combinators.

\begin{code}
      "lazyk" -> [
        3, 0x40,  -- Lazy K loop
        -- BX = IP
        getlocal, ip, setlocal, bx,
        -- [HP] = IP
        getlocal, hp, getlocal, ip, i32store, 2, 0,
        -- [HP + 4] = Var "k"
        getlocal, hp, i32const, 4, i32add, i32const, 128 - 3, i32store, 2, 0,
        -- [HP + 8] = HP
        getlocal, hp, i32const, 8, i32add, getlocal, hp, i32store, 2, 0,
        -- [HP + 12] = Var "+"
        getlocal, hp, i32const, 12, i32add, i32const, 128 - 2, i32store, 2, 0,
        -- [HP + 16] = HP + 8
        getlocal, hp, i32const, 16, i32add, getlocal, hp, i32const, 8, i32add,
        i32store, 2, 0,
        -- [HP + 20] = Var "0"
        getlocal, hp, i32const, 20, i32add, i32const, 128 - 1, i32store, 2, 0,
        -- IP = HP + 16
        -- HP = HP + 24
        getlocal, hp, i32const, 16, i32add, setlocal, ip,
        getlocal, hp, i32const, 24, i32add, setlocal, hp]
\end{code}

The meaning of our `0` combinator depends on the language. For Nat, it means
the computation is finished and the AX register contains the integer result.
For the others, it means AX contains the next byte of the output string, so
we call the `f` function, reset AX to zero, then continue with the rest
of the computation. This is especially easy in Crazy L: roughly speaking, the
right fold representation automatically does this for us.

\begin{code}
      _ -> []

    ++ [3, 0x40]  -- loop
    ++ concat (replicate (ccount + 1) [2, 0x40])  -- blocks
    ++ [i32const, 128 - 1, getlocal, ip, i32sub,  -- -1 - IP
    0xe] ++ (ccount:[0..ccount])  -- br_table
    -- end 0
    ++ [0xb] ++ case mode of
-- Zero.
      "lazyk" ->
        [getlocal, ax, i32const, 128, 2, i32ge_u,  -- AX >= 256?
        br_if, ccount + 2,  -- br_if function
        getlocal, ax, 0x10, 0,
        i32const, 0, setlocal, ax,
        -- [HP] = BX
        getlocal, hp, getlocal, bx, i32store, 2, 0,
        -- [HP + 4] = HP + 8
        getlocal, hp, i32const, 4, i32add, getlocal, hp, i32const, 8, i32add,
        i32store, 2, 0,
        -- [HP + 8] = Var "s"
        getlocal, hp, i32const, 8, i32add, i32const, 128 - 4, i32store, 2, 0,
        -- [HP + 12] = Var "k"
        getlocal, hp, i32const, 12, i32add, i32const, 128 - 3, i32store, 2, 0,
        -- IP = HP
        -- HP = HP + 16
        getlocal, hp, setlocal, ip,
        getlocal, hp, i32const, 16, i32add, setlocal, hp,
        br, ccount + 1]  -- br Lazy K loop
      "fussyk" ->
        [getlocal, ax, i32const, 128, 2, i32ge_u,  -- AX >= 256?
        br_if, ccount + 1,  -- br_if function
        getlocal, ax, 0x10, 0,
        i32const, 0, setlocal, ax,
        -- [HP] = BX
        getlocal, hp, getlocal, bx, i32store, 2, 0,
        -- [HP + 4] = Var ">"
        getlocal, hp, i32const, 4, i32add, i32const, 128 - 5, i32store, 2, 0,
        -- IP = HP
        -- HP = HP + 8
        getlocal, hp, setlocal, ip,
        getlocal, hp, i32const, 8, i32add, setlocal, hp,
        br, ccount]  -- br loop
      "crazyl" ->
        [getlocal, ax, 0x10, 0,
        i32const, 0, setlocal, ax,
        -- IP = BX
        getlocal, bx, setlocal, ip,
        br, ccount]  -- br loop
      "nat" ->
        [getlocal, ax, 0x10, 2,
        br, ccount + 1]  -- br function
      _ -> error "unreachable"
\end{code}

The `(+)`, S, and K combinators have the same effects in all languages.

\begin{code}
    ++ [0xb,  -- end 1
-- Successor.
    -- AX = AX + 1
    getlocal, ax, i32const, 1, i32add, setlocal, ax,
    -- IP = [[SP] + 4]
    getlocal, sp, i32load, 2, 0, -- align 2, offset 0.
    i32const, 4, i32add, i32load, 2, 0,
    setlocal, ip,
    -- SP = SP + 4
    getlocal, sp, i32const, 4, i32add, setlocal, sp,
    br, ccount - 1,  -- br loop
    0xb,  -- end 2
-- K combinator.
    -- IP = [[SP] + 4]
    getlocal, sp, i32load, 2, 0, i32const, 4, i32add, i32load, 2, 0,
    setlocal, ip,
    -- SP = SP + 8
    getlocal, sp, i32const, 8, i32add, setlocal, sp,
    br, ccount - 2,  -- br loop
    0xb,  -- end 3
-- S combinator.
    -- [HP] = [[SP] + 4]
    getlocal, hp,
    getlocal, sp, i32load, 2, 0, i32const, 4, i32add, i32load, 2, 0,
    i32store, 2, 0,
    -- [HP + 4] = [[SP + 8] + 4]
    getlocal, hp, i32const, 4, i32add,
    getlocal, sp, i32const, 8, i32add, i32load, 2, 0,
    i32const, 4, i32add, i32load, 2, 0,
    i32store, 2, 0,
    -- [HP + 8] = [[SP + 4] + 4]
    getlocal, hp, i32const, 8, i32add,
    getlocal, sp, i32const, 4, i32add, i32load, 2, 0,
    i32const, 4, i32add, i32load, 2, 0,
    i32store, 2, 0,
    -- [HP + 12] = [HP + 4]
    getlocal, hp, i32const, 12, i32add,
    getlocal, hp, i32const, 4, i32add, i32load, 2, 0,
    i32store, 2, 0,
    -- SP = SP + 8
    -- [[SP]] = HP
    getlocal, sp, i32const, 8, i32add, teelocal, sp,
    i32load, 2, 0,
    getlocal, hp,
    i32store, 2, 0,
    -- [[SP] + 4] = HP + 8
    getlocal, sp, i32load, 2, 0, i32const, 4, i32add,
    getlocal, hp, i32const, 8, i32add,
    i32store, 2, 0,
    -- IP = HP
    -- HP = HP + 16
    getlocal, hp, teelocal, ip,
    i32const, 16, i32add, setlocal, hp,
    br, ccount - 3,  -- br loop
    0xb,  -- end 4
\end{code}

Because we use BX differently for Fussy K and Crazy L, it turns out we can
combine our `(!)` and `(>)` combinators here. In both cases, we prepare to
reduce `x(+)0` where `x` is the first argument and set BX to the second
argument. In our implementation for `0` above, BX is handled according to the
language selected.

\begin{code}
-- ">": Fussy K / Crazy L.
    -- [HP] = [[SP] + 4]
    getlocal, hp, getlocal, sp, i32load, 2, 0, i32const, 4, i32add,
    i32load, 2, 0, i32store, 2, 0,
    -- [HP + 4] = Var "+"
    getlocal, hp, i32const, 4, i32add, i32const, 128 - 2, i32store, 2, 0,
    -- [HP + 8] = HP
    getlocal, hp, i32const, 8, i32add, getlocal, hp, i32store, 2, 0,
    -- [HP + 12] = Var "0"
    getlocal, hp, i32const, 12, i32add, i32const, 128 - 1, i32store, 2, 0,
    -- IP = HP + 8
    getlocal, hp, i32const, 8, i32add, setlocal, ip,
    -- BX = [[SP + 4] + 4]
    getlocal, sp, i32const, 4, i32add, i32load, 2, 0,
    i32const, 4, i32add, i32load, 2, 0, setlocal, bx,
    -- HP = HP + 16
    getlocal, hp, i32const, 16, i32add, setlocal, hp,
    -- SP = SP + 8
    getlocal, sp, i32const, 8, i32add, setlocal, sp,
    br, ccount - 4,  -- br loop
    0xb,  -- end 5
-- ".": Crazy L nil function.
    br, 2,  -- br function
    0xb,  -- end 6
\end{code}

Function application is more complex this time, as we must watch out for the
streaming input `(<)` combinator:

\begin{code}
-- Application.
    -- SP = SP - 4
    -- [SP] = IP
    getlocal, sp, i32const, 4, i32sub,
    teelocal, sp, getlocal, ip, i32store, 2, 0,

    -- [IP] = Var "<"?
    2, 0x40,  -- block <
    getlocal, ip, i32load, 2, 0, i32const, 128 - 8, i32ne,
    br_if, 0,
    -- [HP] = vireo or foldBird
    getlocal, hp, i32const]
    ++ leb128 (if mode == "crazyl" then addrRFold else addrVireo) ++
    [i32store, 2, 0,
    -- [HP + 4] = getChar * 8
    getlocal, hp, i32const, 4, i32add, 0x10, 1, i32const, 8, i32mul,
    i32store, 2, 0] ++ (if mode /= "crazyl" then [] else
      -- [HP + 4] = 256 * 8?
      [2, 0x40,  -- block Crazy L nil
      getlocal, hp, i32const, 4, i32add,
      i32load, 2, 0, i32const, 128, 16, i32lt_u,
      br_if, 0,  -- br Crazy L nil
      -- [IP] = Var "S"
      getlocal, ip, i32const, 128 - 4, i32store, 2, 0,
      -- [IP + 4] = Var "K"
      getlocal, ip, i32const, 4, i32add, i32const, 128 - 3, i32store, 2, 0,
      br, 1,     -- br <
      0xb])      -- end Crazy L nil
    -- [HP + 8] = Var "<"
    ++ [getlocal, hp, i32const, 8, i32add, i32const, 128 - 8, i32store, 2, 0,
    -- [IP] = HP
    getlocal, ip, getlocal, hp, i32store, 2, 0,
    -- [IP + 4] = HP + 8
    getlocal, ip, i32const, 4, i32add,
    getlocal, hp, i32const, 8, i32add, i32store, 2, 0,
    -- HP = HP + 16
    getlocal, hp, i32const, 16, i32add, setlocal, hp,
    0xb,      -- end <

    -- IP = [IP]
    getlocal, ip, i32load, 2, 0, setlocal, ip,
    br, 0,  -- br loop
    0xb]    -- end loop
    ++ case mode of
      "lazyk" -> [0xb]    -- end Lazy K loop
      _       -> []
    ++ [0xb]],  -- end function

  -- Data section.
  sect 11 [[0, i32const, 0, 0xb] ++ lenc heap]]
  where
    heap = encodeTree $ case mode of 
      "lazyk"  -> e :@ (Var "<" :@ Var "<")
      "fussyk" -> e :@ (Var "<" :@ Var "<") :@ Var ">"
      "crazyl" -> e :@ (Var "<" :@ Var "<") :@ Var ">" :@ Var "."
      "nat"    -> e :@ Var "+" :@ Var "0"
\end{code}

== Web UI ==

We conclude by connecting buttons and textboxes with code.

\begin{code}
main = withElems ["source", "input", "output", "sk", "asm", "compB", "runB"] $
    \[sEl, iEl, oEl, skEl, aEl, compB, runB] -> do
  inp <- newMVar ""
  let
    putChar :: Int -> IO ()
    putChar c = do
      v <- getProp oEl "value"
      setProp oEl "value" $ v ++ [chr c]
    putInt :: Int -> IO ()
    putInt n = setProp oEl "value" $ show n
    getChar :: IO Int
    getChar = do
      s <- takeMVar inp
      case s of
        [] -> do
          putMVar inp []
          pure 256
        (h:t) -> do
          putMVar inp t
          pure $ ord h
  export "putChar" putChar
  export "putInt"  putInt
  export "getChar" getChar
  let
    setupDemo name s = do
      Just b <- elemById $ name ++ "B"
      Just d <- elemById $ name ++ "Demo"
      Just r <- elemById $ name ++ "Radio"
      b `onEvent` Click $ const $ do
        setProp sEl "value" =<< getProp d "value"
        setProp r "checked" "true"
        setProp iEl "value" s
        setProp oEl "value" ""
  setupDemo "nat" ""
  setupDemo "lazyk" "gateman"
  setupDemo "fussyk" "(ignored)"
  setupDemo "crazyl" "length"
  compB `onEvent` Click $ const $ do
    setProp skEl "value" ""
    setProp aEl "value" ""
    s <- getProp sEl "value"
    case parseProgram s of
      Left err -> setProp skEl "value" $ "error: " ++ show err
      Right sk -> do
        let
          f s = do
            Just el <- elemById $ s ++ "Radio"
            bool "" s . ("true" ==) <$> getProp el "checked"
        lang <- concat <$> mapM f ["nat", "lazyk", "fussyk", "crazyl"]
        let asm = compile lang sk
        setProp skEl "value" $ show sk
        setProp aEl "value" $ show asm
  runB `onEvent` Click $ const $ do
    setProp oEl "value" ""
    s <- getProp iEl "value"
    _ <- takeMVar inp
    putMVar inp s
    asmStr <- getProp aEl "value"
    let asm = read asmStr :: [Int]
    ffi "runWasmInts" asm :: IO ()
#endif
\end{code}
