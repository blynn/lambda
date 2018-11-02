== Crazy L ==

Let's build a better compiler based on combinators.
This time, we'll produce WebAssembly for a family of languages related to
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
<button id="revB">Yum!</button>
<button id="sortB">Sort</button>
<br>
<textarea id="revDemo" hidden>
\l.l(\htx.t(\cn.ch(xcn)))i(sk)
</textarea>
<textarea id="sortDemo" hidden># Sort a list in Crazy L
N=sk
z=\n.n(\x.N)k
V=\xyf.fxy
c=\htcn.ch(tcn)
P=\nfx.n(\gh.h(gf))(\u.x)(\u.u)
L=\mn.(\pq.pqp)(z(nPm))((\pab.pba)(z(mPn)))
H=\l.l(\ht.h)N
u=\l.l(\ht.k)N
f=\xp.u(pk)(L(H(pk))x(V(pk)(cx(pN)))(VN(cx(c(H(pk))(pN)))))(VN(cx(pN)))
r=\xl.(\q.(u(qk)(c(H(qk))(qN))(qN)))(lf(V(cxN)N))
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
\l.c(l(ku)z)(sk)
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
Vireo, or the pairing bird, so we can write $\iota = V S K$. Then

\[
\iota \iota = \iota S K = S S K K = S K (K K) = I
\]

from which we deduce $\iota (\iota \iota) = S K$,
$\iota (\iota (\iota \iota)) = K$, and
$\iota (\iota (\iota (\iota \iota))) = S$.

https://esolangs.org/wiki/Jot[The Jot language] is another two-symbol
language with an interesting property: any string of 0s and 1s is a valid
program:

\[
\begin{align}
[]   &= I \\
[F0] &= \iota [F] = [F]S K \\
[F1] &= \lambda x y.[F](x y) = S(K[F])
\end{align}
\]

Here $[F]$ represents the decoding of a string $F$ of 0s and 1s. In
particular, the empty string is a valid program: it represents $I$, the
identity combinator.

Incidentally, https://en.wikipedia.org/wiki/Iota_and_Jot[the description of Jot
on Wikipedia seems erroneous] (as of May 2017). I get the impression that
$\iota = \lambda w.w S K$ is confused with $\lambda w.S(K w)$,
so while $w0$ indeed denotes $\iota w$, $w1$ actually denotes $S(K w)$.

Also, in general, $0^* w$ differs from $I w$, which is only a minor issue for
Gödel numbering: like floating point numbers, we can tacitly assume the
presence of a leading 1 bit. All the same, there must be some reason for
decoding from the end of the string instead of the beginning, and it would be
nice if leading zeroes could be omitted without changing a program's meaning.

In sum, we can express SK terms in various languages as follows:

------------------------------------------------------------------------------
dumpIota     = dumpWith '*' "*i*i*ii" "*i*i*i*ii"
dumpJot      = dumpWith '1' "11100"   "11111000"
dumpUnlambda = dumpWith '`' "k"       "s"

dumpWith apCh kStr sStr = fix $ \f -> \case
  x :@ y  -> apCh:f x ++ f y
  Var "K" -> kStr
  Var "S" -> sStr
  _       -> error "SK terms only"
------------------------------------------------------------------------------

== Lazy K ==

https://tromp.github.io/cl/lazy-k.html[Lazy K] combines the syntaxes of SKI
combinator calculus, Unlambda, Iota, and Jot, which amazingly coexist mostly in
peace. The only exception is the `i` program, which Lazy K interprets as the
identity combinator rather than the iota combinator.

Lazy K expects the first and only argument of the given program to be a list,
in the form of nested Church-encoded pairs. The end of a finite list is
represented by an infinite list where every element is (the Church encoding of)
256. For example, the string "AB" would be represented as:

------------------------------------------------------------------------------
V 65 (V 66 (V 256 (V 256 (...))))
------------------------------------------------------------------------------

where, as before, `Vxyz = zxy`, and the numbers are Church-encoded. The
reference interpreter treats any number above 256 as 256.

This is an unfortunate choice. Lambdas and combinators hail from a beautiful
mathematical world, which Lazy K has polluted with some constant or other.
Obviously, the constant 256 was chosen to suit certain real-life situations,
but why constrain ourselves so early in the design process?

Better to represent the end of the list out-of-band. Then we could operate on
lists of arbitrary natural numbers, as well as the case when the input is a
list of 8-bit bytes. When it's time to write interpreters and compilers, we
may impose limits due to the messiness of the real world, but languages
themselves ought to be neat.

== Fussy K ==

The reference implementation of Lazy K is sloppy with respect to the output.
Ideally, it should look for `V 256 x` in the output list for any value of `x`,
at which point the program should terminate, but instead, the current item of
the list is tested by applying it to the K combinator, and if this returns 256
then the program halts. Indeed, the documentation explicitly mentions that `K
256` is a valid end-of-output marker.

However `V 256 x` behaves differently to `K 256`. For example
`V 256 x (KI) = x` while `K 256 (KI) = 256`. This complicates our
implementation.

We tie up this loose end by defining Fussy K to be the Lazy K language as it
is specified, that is, the output list must be terminated with a 256 in the
first argument of a V combinator; `K 256` will not do.

== Crazy L ==

Let's design a cleaner Lazy K, and add a few features.

For the input encoding, instead of pairs, we use
https://en.wikipedia.org/wiki/Church_encoding#Represent_the_list_using_right_fold[the right fold representation of lists].

List manipulations become elegant. With types, we could readily prove certain
programs terminate on finite inputs, and other theorems. Also for finite
inputs, we could choose any evaluation order when running our program.
Nonetheless, we'll stick with lazy evaluation so we can also handle infinite
inputs.

We write our interpreter and compiler to expect right fold encodings, and
use the following shim to convert a list `x` to Lazy K's input encoding:

------------------------------------------------------------------------------
\x.xV(Y(\f.V 256 f))
------------------------------------------------------------------------------

where Y is the Y combinator and 256 is the Church encoding of 256.

We add support for lambda abstractions and top-level definitions, where all
variables must be single characters other than `skiSICKB`.

We name our language Crazy L.

== Parsing ==

We catch up with an old friend: an AST for lambda calculus terms. Once again,
we wish to eliminate all the lambda abstractions, leaving only variables and
applications.

\begin{code}
{-# LANGUAGE CPP #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FlexibleInstances #-}
#ifdef __HASTE__
{-# LANGUAGE OverloadedStrings #-}
import Haste.DOM
import Haste.Events
import Haste.Foreign
import Data.Bool
import Data.IORef
#else
import Data.Function (fix)
import System.Console.Haskeline
import System.Environment
import System.IO
import Test.HUnit
import Criterion.Main hiding (env)
#endif
import Control.Monad
import Data.Char
import Data.List
import qualified Data.Map as M
import Data.Map (Map, (!))
import Text.ParserCombinators.Parsec

infixl 5 :@
data Term = Var String | Term :@ Term | Lam String Term
\end{code}

We define a few combinators. Some of their names are a nod to Smullyan's "To
Mock a Mockingbird".

\begin{code}
consBird, succBird, vireo, skk, vsk, forever256 :: Term
consBird = mustParse "((BS)((B(BB))(CI)))"
succBird = Var "S" :@ Var "B"
vireo = Var "B" :@ Var "C" :@ (Var "C" :@ Var "I")
skk = Var "I"
vsk = vireo :@ Var "S" :@ Var "K"
forever256 = mustParse "SII(B(BC(CI)(SII(SII(SBI))))(SII))"
\end{code}

Our parser closely follows https://tromp.github.io/cl/lazy-k.html[the grammar
specified in the description of Lazy K]. Differences:

 * We support lambda abstractions, e.g: `\x.x`
 * With `=`, we can assign terms to any letter except those in `"skiSICKB"`,
 e.g: `c=\htcn.ch(tcn)`. The letters `B` and `C` are reserved for the B and C
 combinators.

Definitions may only use the core language and previously defined letters. In
particular, recursive functions must be defined via the Y combinator.

We expect a term at the top-level, which we consider to be a definition of
the special symbol `main`.

Comments begin with `#` and all whitespace except newlines are ignored.

Later definitions override earlier ones. In particular, because we support Jot,
any trailing newlines (possibly with comments) are significant and change the
program to be simply the I combinator.

\begin{code}
top :: Parser (String, Term)
top = (try super <|> (,) "main" <$> ccexpr) <* eof where
  super = (,) <$> var <*> (char '=' >> ccexpr)
  ccexpr   = option skk $ foldl1 (:@) <$> many1 expr
  expr     = const skk <$> char 'i' <|> expr'
  iotaexpr = const vsk <$> char 'i' <|> expr'
  expr' = jotRev . reverse <$> many1 (oneOf "01")
      <|> const skk <$> char 'I'
      <|> Var . pure . toUpper <$> oneOf "ks"
      <|> Var . pure <$> letter
      <|> between (char '(') (char ')') ccexpr
      <|> (char '`' >> (:@) <$> expr <*> expr)
      <|> (char '*' >> (:@) <$> iotaexpr <*> iotaexpr)
      <|> flip (foldr Lam) <$> between (char '\\' <|> char '\0955') (char '.')
        (many1 var) <*> ccexpr

  var = lookAhead (noneOf "skiSICKB") >> pure <$> letter

  jotRev []       = skk
  jotRev ('0':js) = jotRev js :@ Var "S" :@ Var "K"
  jotRev ('1':js) = Var "S" :@ (Var "K" :@ jotRev js)
  jotRev _        = error "bad Jot term"

parseLine :: String -> Either ParseError (String, Term)
parseLine = parse top "" . filter (not . isSpace) . takeWhile (/= '#')

mustParse :: String -> Term
mustParse = either undefined snd . parseLine
\end{code}

Since a definition may only use previously defined symbols, we can substitute
letters for their definitions terms as we parse a program line by line, and
keep our terms fully expanded and bracket abstracted in `env`.

\begin{code}
sub :: [String] -> [(String, Term)] -> Term -> Either String Term
sub bvs env = \case
  x :@ y -> (:@) <$> sub bvs env x <*> sub bvs env y
  Var s | elem s bvs             -> Right $ Var s
        | [c] <- s, elem c combs -> Right $ Var s
        | Just t <- lookup s env -> Right t
        | otherwise              -> Left $ s <> " is free"
  Lam s t -> Lam s <$> sub (s:bvs) env t

parseEnv :: [(String, Term)] -> String -> Either String [(String, Term)]
parseEnv env ln = case parseLine ln of
  Left e -> Left $ show e
  Right (s, t) -> case sub [] env t of
    Right u -> Right $ (s, u):env
    Left e -> Left e

parseProgram :: String -> Either String Term
parseProgram program = case foldM parseEnv [] $ lines program of
  Left err -> Left err
  Right env -> maybe (Left "missing main") Right $ lookup "main" env
\end{code}

A program is interpreted according to which language variant we've chosen.
Lazy K, Fussy K, and Crazy L all take streams of bytes as input and all produce
streams of bytes as output. To this, we add the Nat language, we expects
no input and just outputs a Church-encoded number.

\begin{code}
data Lang = LazyK | FussyK | CrazyL | Nat
\end{code}

== De Bruijn indices ==

We shall need
https://en.wikipedia.org/wiki/De_Bruijn_index['De Bruijn indices]', that is,
we replace each variable with an integer representing the number of `Lam` nodes
we encounter as we travel up the parse tree before reaching the binding
abstraction. (This is similar to wasm branch labels!)

For example,

\[
\lambda f.(\lambda x.x x)(\lambda x.f(x x))
\]

becomes:

\[
\lambda(\lambda 0 0)(\lambda 1(0 0))
\]

For example, in De Bruijn notation, $S = \lambda\lambda\lambda 2 0(1 0)$
and $K = \lambda\lambda 1$.

We employ http://okmij.org/ftp/tagless-final/[a tagless final representation]
for De Bruijn terms:

\begin{code}
infixl 5 #
class Deb repr where
  ze :: repr
  su :: repr -> repr
  lam :: repr -> repr
  (#) :: repr -> repr -> repr
  prim :: String -> repr
\end{code}

We declare an instance so we can display De Bruijn terms for debugging and
testing.

\begin{code}
data Out = Out { unOut :: String }
instance Deb Out where
  ze = Out "Z"
  su e = Out $ "S(" <> unOut e <> ")"
  lam e = Out $ "^" <> unOut e
  e1 # e2 = Out $ unOut e1 <> unOut e2
  prim s = Out s
\end{code}

== Sick B ==

This time, we need the B and C combinators (`(.)` and `flip` in Haskell) as
well as the S and K combinators. We also directly implement I combinators,
rather than making do with SKK.

A straightforward recursion computes the De Bruijn indices.
Because we may be asked to perform bracket abstraction on the output of our
bracket abstraction routine, we must support B and C when converting to De
Bruijn indices.

\begin{code}
toDeb :: Deb repr => [String] -> Term -> repr
toDeb env = \case
  Var s -> case elemIndex s env of
    Nothing -> case s of
      "S" -> lam $ lam $ lam $ su(su ze) # ze # (su ze # ze)
      "B" -> lam $ lam $ lam $ su(su ze)      # (su ze # ze)
      "C" -> lam $ lam $ lam $ su(su ze) # ze #  su ze
      "K" -> lam $ lam $ su ze
      "I" -> lam ze
      _   -> prim s
    Just n -> iterate su ze !! n
  Lam s t -> lam $ toDeb (s:env) t
  x :@ y -> toDeb env x # toDeb env y
\end{code}

Now we can apply http://okmij.org/ftp/tagless-final/ski.pdf[a powerful bracket
abstraction algorithm due to Oleg Kiselyov]. Again we use a tagless final
representation for the output of the algorithm.

\begin{code}
infixl 5 ##
class SickB repr where
  kV :: String -> repr
  (##) :: repr -> repr -> repr

instance SickB (Bool -> Out) where
  kV s _ = Out s
  (e1 ## e2) False = Out $        unOut (e1 False) <> unOut (e2 True)
  (e1 ## e2) _     = Out $ "(" <> unOut (e1 False) <> unOut (e2 True) <> ")"
\end{code}

We introduce another data type because the algorithm needs to distinguish
between closed terms, and several kinds of unclosed terms. We return a closed
term, but along the way we manipulate open terms.

Kiselyov's code omits the `(V, V)` case because it applies to a simply typed
algebra. We allow untyped terms.

\begin{code}
data Oleg repr = C {unC :: repr} | N (Oleg repr) | W (Oleg repr) | V
instance SickB repr => Deb (Oleg repr) where
  prim s = C (kV s)
  ze = V
  su = W
  l # r = case (l, r) of
    (W e, V) -> N e
    (V, W e) -> N $ C (c ## i) # e
    (N e, V) -> N $ C s # e # C i
    (V, N e) -> N $ C (s ## i) # e
    (C d, V) -> N $ C d
    (V, C d) -> N $ C $ c ## i ## d
    (V, V)   -> N $ C $ s ## i ## i

    (W e1, W e2) -> W $ e1 # e2
    (W e, C d)   -> W $ e # C d
    (C d, W e)   -> W $ C d # e
    (W e1, N e2) -> N $ C b # e1 # e2
    (N e1, W e2) -> N $ C c # e1 # e2
    (N e1, N e2) -> N $ C s # e1 # e2
    (C d, N e)   -> N $ C (b ## d) # e
    (N e, C d)   -> N $ C (c ## c ## d) # e
    (C d1, C d2) -> C $ d1 ## d2
    where [s,i,c,b] = kV . pure <$> "SICB"
  lam = \case
    V   -> C i
    C d -> C $ k ## d
    N e -> e
    W e -> C k # e
    where [i,k] = kV . pure <$> "IK"

showBabs :: Term -> String
showBabs t = unOut $ unC (toDeb [] t) False
\end{code}

== Interpreter ==

We build an interpreter to guide our compiler design.

We envision a machine with 4 registers (you can tell I grew up on x86 assembly):

 * IP: instruction pointer that also holds instructions
 * SP: stack pointer, growing downwards from the top of memory.
 * HP: heap pointer, growing upwards from the bottom of memory.
 * AX: accumulator

We arbitrarily decide our wasm instances will request 64 pages of memory.

\begin{code}
pageCount :: Int
pageCount = 64

maxSP :: Int
maxSP = pageCount * 65536

data VM = VM
  { ip, hp, sp :: Int
  , ax :: Int
  , mem :: Map Int Int
  , input :: String
  , lang :: Lang
  }
\end{code}

The SICKB combinators are standard. We introduce special combinators to deal
with the real world.

\begin{code}
combs :: [Char]
combs = "SICKB0+<>."
\end{code}

The heap is organized as an array of 8-byte entries, each consisting of two
4-byte combinators `x` and `y`. The meaning of such an entry is `xy`.

A negative 4-byte value represents one of the primitive combinators.
Otherwise it is the address of another 8-byte entry in the heap.

This encoding scheme means if a term consists of a single primitive combinator,
such as K, then we must represent it as IK since at minimum a cell holds two
combinators.

\begin{code}
instance SickB (Int -> [Int]) where
  kV s _ = [enCom s]
  (e1 ## e2) n = n:h1:h2:t1++t2 where
    (h1:t1) = e1 (n + 8)
    (h2:t2) = e2 (n + 8 + wlen t1)

wlen :: [a] -> Int
wlen = (4*) . length

enCom :: String -> Int
enCom [c] | Just n <- elemIndex c combs = -n - 1
enCom s = error $ show s

encAt :: Int -> Term -> [Int]
encAt n t = tail $ unC (toDeb [] t) n

dump :: VM -> String
dump VM{..} = unlines $ take 50 . f <$> ps where
  f a | a < 0 = pure $ combs!!(-a - 1)
  f a         = "(" ++ f (de a) ++ f (de $ a + 4) ++ ")"
  ps = ip:[de $ 4 + de p | p <- [sp, sp + 4..maxSP - 4]]
  de k | Just v <- M.lookup k mem = v
       | otherwise = error $ "bad deref: " ++ show k
\end{code}

We place the Church encoded integers from [0..256] in linear memory starting
from 0. Each takes one 8-byte cell, so that the Church encoding of 'n' lies at
address '8n' in memory.

Our input handler uses these to quickly map a number up to 256 to its Church
encoding. Larger input numbers are unsupported. In principle, we could generate
encodings for them on demand, but if we really wanted big numbers we'd use a
more efficient encoding, or add a primitive integer type.

Zero is represented by SK, and 'n + 1' is represented by 'm n' where 'm' is
the combinator that computes the successor of a Church number. We place the
definition of 'm' just after the Church numbers, that is, at memory address
`8*257`.

\begin{code}
gen :: [Int]
gen = enCom "S" : enCom "K" :           -- Zero
  concat [[m, 8*n] | n <- [0..255]] ++  -- [1..256]
  encAt m succBird                      -- Successor combinator.
  where m = 8*257
\end{code}

We encode a program immediately after the above. We add our special combinators
differently for each language so that the term will behave accordingly, which
we explain later.

\begin{code}
encodeTerm :: Lang -> Term -> [Int]
encodeTerm lang t = (gen ++) $ encAt (wlen gen) $ case lang of
  Nat     -> t :@ Var "+" :@ Var "0"
  LazyK   -> t :@ ugh :@ Var ">" :@ Var "+" :@ Var "0"
  FussyK  -> t :@ ugh :@ Var ">"
  CrazyL  -> t :@ inp :@ Var ">" :@ Var "."
  where
  inp = Var "<" :@ consBird
  ugh = inp :@ vireo :@ forever256
\end{code}

The IP register points to our term, which is just after the Church-encoded
integers. The HP register points to the free heap, which begins just after our
program. The SP register points to the top of memory, as the stack is
initially empty.

\begin{code}
sim :: Lang -> String -> Term -> String
sim mode inp e = exec VM
  { ip = wlen gen
  , sp = maxSP
  , hp = wlen bs
  , ax = 0
  , mem = M.fromList $ zip [0,4..] bs
  , input = inp
  , lang = mode
  } where
  bs = encodeTerm mode e
\end{code}

Executing a combinator involves a few subtleties. Lazy evaluation is
important for the S, B, and C combinators, that is, we must memoize so future
evaluations avoid recomputing the same reduction. Without this, even simple
programs may be too slow.

We also memoize the result of the K combinator, but this is less vital.

There may be some memoization possible with the I combinator, but it seems
similar to a tag to me, so this may be unimportant.

The `upd` function updates the heap entry that the top of the stack refers to,
as well as the IP register. It powers memoization and lazy input.

\begin{code}
upd :: Int -> Int -> VM -> VM
upd a b vm@VM{..} = setIP a $ vm
  { mem = M.insert (mem!sp) a $ M.insert ((mem!sp) + 4) b mem }

exec :: VM -> String
exec vm@VM{..} | ip < 0 = case combs!!(-ip - 1) of
  'S' -> rec $ upd hp (hp + 8) . pop 2 . putHP [arg 0, arg 2, arg 1, arg 2, hp, hp + 8]
  'I' -> rec $ setIP (arg 0) . pop 1
  'C' -> rec $ putHP [arg 0, arg 2] . upd hp (arg 1) . pop 2
  -- Unmemoized: 'K' -> rec $ setIP (arg 0) . pop 2
  'K' -> rec $ upd (enCom "I") (arg 0) . pop 1
  'B' -> rec $ putHP [arg 1, arg 2] . upd (arg 0) hp . pop 2
\end{code}

The meaning of our special combinators depends on the language.

For Nat, the `(+)` combinator acts like I except it also increments AX and the
`0` combinator outputs AX and terminates. Then given a Nat program `t`, we
recover its output by reducing `t(+)0`, since its output is Church-encoded.

In theory, `exec` should return values, the `0` combinator should return the
integer 0, and `(+)` should increment its first argument and return it.
(Ultimately some outer function would print the result of `exec`.)
But since we assume `t` evaluates to a Church number, and since we control the
evaluation order, we optimize by giving certain side effects to these two
combinators. We cheat similarly with the other languages.

The `(<)` combinator is always applied to some combinator `x`. Then when we
reach it during evaluation, we know the top entry of the stack is `(<)x`.
We set `x` to be the cons combinator for right-fold representations of lists,
and we replace the entry with `xn(<0)` where `n` is the Church encoding of the
next byte of input, or `SK` if there is no more input.

The `(>)` combinator is `\xy.x(+)(0y)`. By tweaking how `0` works for the
other languages, we cause this term to turn the first argument (which should
be a Church number) into a byte which we emit, then recurse on `y`.

Lazy K is fiddly because we must handle the case when it skips over our `(>)`
combinator, such as for the program `K(K(256))`. (Normally we supply an
argument to `0` for non-Nat programs to ensure IP == [[SP]] when we evaluate
`0`, but we can skip it for the special-case Lazy K `0` because we know it
should terminate if it gets evaluated.)

\begin{code}
  '0' -> case lang of
    Nat     -> show ax
    CrazyL  -> chr ax : rec (setIP (arg 0) . setAX 0 . pop 1)
    FussyK  -> if ax == 256 then "" else
      chr ax : rec (upd (arg 0) (enCom ">") . setAX 0)
    LazyK   -> if ax >= 256 then "" else
      chr ax : rec (upd (hp + 8) (enCom "0") . putHP
        [arg 0, enCom ">", hp, enCom "+"] . setAX 0)
  -- I combinator with side effect.
  '+' -> rec $ setIP (arg 0) . pop 1 . setAX (ax + 1)
  '>' -> rec $ upd hp (hp + 8) . pop 1 . putHP
    [arg 0, enCom "+", enCom "0", arg 1]
  '.' -> ""
  -- Lazy input. If we reach here, then IP == [[SP]].
  '<' -> case input of
     (h:t) | ord h <= 256 -> exec $ putHP [arg 0, ord h * 8, enCom "<", arg 0] $
        upd hp (hp + 8) vm { input = t }
           | otherwise    -> error "no support for integers > 256"
     _     -> rec $ upd (enCom "S") (enCom "K")
  _ -> error $ "bad combinator\n" ++ dump vm
  where
  rec f = exec $ f vm
  arg n = mem ! (mem ! (sp + n * 4) + 4)
  setAX a v = v {ax = a}
exec vm@VM{..} = exec $ checkOverflow $ vm { sp = sp - 4, mem = M.insert (sp - 4) ip mem, ip = mem ! ip }

pop :: Int -> VM -> VM
pop n vm@VM{..} = vm { sp = sp + 4*n }

setIP :: Int -> VM -> VM
setIP a v = v {ip = a}

putHP :: [Int] -> VM -> VM
putHP as vm@VM{..} = checkOverflow $ vm
 { mem = M.union (M.fromList $ zip [hp, hp + 4..] as) mem, hp = hp + wlen as }

checkOverflow :: VM -> VM
checkOverflow vm@VM{..} | hp >= sp  = error "overflow"
                        | otherwise = vm
\end{code}

== Compiler ==

We have a three import functions this time:

  * The program outputs numbers via `f`.
  * The program calls `g` to get the next input number. This function should
    return a negative number if there is no more input.
  * The Nat language calls `h` to return a 32-bit number.

\begin{code}
leb128 :: Int -> [Int]
leb128 n | n < 64   = [n]
         | n < 128  = [128 + n, 0]
         | otherwise = 128 + (n `mod` 128) : leb128 (n `div` 128)

i32 :: Int
i32 = 0x7f

i32const :: Int
i32const = 0x41

compile :: Lang -> Term -> [Int]
compile mode e = concat
  [ [0, 0x61, 0x73, 0x6d, 1, 0, 0, 0]  -- Magic string, version.
  -- Type section.
  , sect 1 [encSig [i32] [], encSig [] [], encSig [] [i32]]
  -- Import section.
  , sect 2 [
    -- [0, 0] = external_kind Function, type index 0.
    encStr "i" ++ encStr "f" ++ [0, 0],
    -- [0, 2] = external_kind Function, type index 2.
    encStr "i" ++ encStr "g" ++ [0, 2],
    encStr "i" ++ encStr "h" ++ [0, 0]]
  -- Function section. [1] = Type index.
  , sect 3 [[1]]
  -- Memory section. 0 = no-maximum
  , sect 5 [[0, pageCount]]
  -- Export section.
  -- [0, 3] = external_kind Function, function index 3.
  , sect 7 [encStr "e" ++ [0, 3]]
  -- Code section.
  , sect 10 [lenc $ codeSection mode $ length heap]
  -- Data section.
  , sect 11 [[0, i32const, 0, 0xb] ++ lenc heap]] where
  heap = concatMap quad $ encodeTerm mode e
  sect t xs = t : lenc (leb128 (length xs) ++ concat xs)
  -- 0x60 = Function type.
  encSig ins outs = 0x60 : lenc ins ++ lenc outs
  encStr s = lenc $ ord <$> s
  lenc xs = leb128 (length xs) ++ xs
  quad n | n < 0     = [256 + n, 255, 255, 255]
         | otherwise = take 4 $ byteMe n
  byteMe n | n < 256   = n : repeat 0
           | otherwise = n `mod` 256 : byteMe (n `div` 256)
\end{code}

We translate our interpreter into WebAssembly for our compiler.

Our `asmCase` helper deals with the branch numbers for each case in the
`br_table`.

\begin{code}
codeSection :: Lang -> Int -> [Int]
codeSection mode heapEnd =
  [1, 4, i32,
  i32const] ++ leb128 (wlen gen) ++ [setlocal, ip,
  i32const] ++ leb128 maxSP ++ [setlocal, sp,
  i32const] ++ leb128 heapEnd ++ [setlocal, hp,
  3, 0x40]  -- loop
  ++ concat (replicate (ccount + 1) [2, 0x40])  -- blocks
  ++ [i32const, 128 - 1, getlocal, ip, i32sub,  -- -1 - IP
  br_table] ++ (ccount:[0..ccount])  -- br_table
  ++ [0xb] ++ concat (zipWith asmCase [0..] combs)
\end{code}

Function application walks down the tree to find the combinator to run next, and
builds up a spine on the stack as it goes.

\begin{code}
  -- Application is the default case.
  -- SP = SP - 4
  -- [SP] = IP
  ++ [getlocal, sp, i32const, 4, i32sub, teelocal, sp,
  getlocal, ip, i32store, 2, 0,
  -- IP = [IP]
  getlocal, ip, i32load, 2, 0, setlocal, ip,
  br, 0,  -- br loop
  0xb,    -- end loop
  0xb]  -- end function
  where
  br       = 0xc
  br_if    = 0xd
  br_table = 0xe
  getlocal = 0x20
  setlocal = 0x21
  teelocal = 0x22
  i32load  = 0x28
  i32store = 0x36
  i32ge_s  = 0x4e
  i32ge_u  = 0x4f
  i32add   = 0x6a
  i32sub   = 0x6b
  i32mul   = 0x6c
  ip = 0  -- instruction pointer, can also hold instructions
  sp = 1  -- stack pointer
  hp = 2  -- heap pointer
  ax = 3  -- accumulator
  ccount = length combs
  asmCase combIndex combName = let
    loopLabel = ccount - combIndex
    exitLabel = loopLabel + 1
    loop = [br, loopLabel]
    asmCom c = [i32const, 128 + enCom c]
    asmIP ops = ops ++ [setlocal, ip]
    asmArg n = [getlocal, sp, i32load, 2, 4*n, i32load, 2, 4]
    asmPop 0 = []
    asmPop n = [getlocal, sp, i32const, 4*n, i32add, setlocal, sp]
    withHeap xs body = concat (zipWith hAlloc xs [0..]) ++ body
      ++ [getlocal, hp, i32const, 4*length xs, i32add, setlocal, hp]
    hAlloc x n = [getlocal, hp] ++ x ++ [i32store, 2, 4*n]
    hNew 0 = [getlocal, hp]
    hNew n = [getlocal, hp, i32const, 8*n, i32add]
    updatePop n x y = concat
      [ [getlocal, sp, i32load, 2, 4*n], x, [teelocal, ip, i32store, 2, 0]
      , [getlocal, sp, i32load, 2, 4*n], y, [i32store, 2, 4]
      , asmPop n
      ]
    in (++ [0xb]) $ case combName of
\end{code}

The following is similar to the `exec` function of our interpreter.

\begin{code}
    '0' -> case mode of
      Nat -> [getlocal, ax, 0x10, 2, br, exitLabel]  -- Print AX.
      LazyK ->
        [getlocal, ax, i32const, 128, 2, i32ge_u,  -- AX >= 256?
        br_if, exitLabel,  -- br_if exit
        getlocal, ax, 0x10, 0,  -- else output AX
        -- AX = 0
        i32const, 0, setlocal, ax
        ] ++ withHeap [asmArg 0, asmCom ">", hNew 0, asmCom "+", asmCom "0", asmCom "."] (updatePop 0 (hNew 1) (hNew 2)) ++ loop
      FussyK ->
        [getlocal, ax, i32const, 128, 2, i32ge_u,  -- AX >= 256?
        br_if, exitLabel,  -- br_if exit
        getlocal, ax, 0x10, 0,  -- else output AX
        -- AX = 0
        i32const, 0, setlocal, ax
        ] ++ updatePop 0 (asmArg 0) (asmCom ">") ++ loop
      CrazyL -> concat
        [ [getlocal, ax, 0x10, 0, i32const, 0, setlocal, ax]
        , asmIP (asmArg 0), asmPop 1, loop]
    '+' -> concat
      [ [getlocal, ax, i32const, 1, i32add, setlocal, ax]  -- AX = AX + 1
      , asmIP (asmArg 0) , asmPop 1 , loop ]
    'K' -> updatePop 1 (asmCom "I") (asmArg 0) ++ loop
    'S' -> withHeap (asmArg <$> [0, 2, 1, 2]) (updatePop 2 (hNew 0) (hNew 1)) ++ loop
    '>' -> withHeap [asmArg 0, asmCom "+", asmCom "0", asmArg 1] (updatePop 1 (hNew 0) (hNew 1)) ++ loop
    '.' -> [br, exitLabel]  -- br exit
    'I' -> concat [asmIP $ asmArg 0, asmPop 1, loop]
    '<' -> concat
      [ [0x10, 1, teelocal, ip]  -- Get next character in IP.
      , [i32const, 0, i32ge_s, 4, 0x40]  -- if >= 0
      , withHeap [asmArg 0, [getlocal, ip, i32const, 8, i32mul], asmCom "<", asmArg 0] (updatePop 0 (hNew 0) (hNew 1))
      , [5]  -- else
      , updatePop 0 (asmCom "S") (asmCom "K")
      , [0xb]  -- end if
      , loop
      ]
    'B' -> withHeap [asmArg 1, asmArg 2] (updatePop 2 (asmArg 0) (hNew 0)) ++ loop
    'C' -> withHeap [asmArg 0, asmArg 2] (updatePop 2 (hNew 0) (asmArg 1)) ++ loop
    e -> error $ "bad combinator: " ++ [e]
\end{code}

== Web UI ==

We conclude by connecting buttons and textboxes with code.

\begin{code}
#ifdef __HASTE__
(<>) = (++)

main :: IO ()
main = withElems ["source", "input", "output", "sk", "asm", "compB", "runB"] $
    \[sEl, iEl, oEl, skEl, aEl, compB, runB] -> do
  inp <- newIORef []
  bin <- newIORef []
  let
    putCh :: Int -> IO ()
    putCh c = do
      v <- getProp oEl "value"
      setProp oEl "value" $ v ++ [chr c]
    putInt :: Int -> IO ()
    putInt n = setProp oEl "value" $ show n
    getCh :: IO Int
    getCh = do
      s <- readIORef inp
      case s of
        [] -> pure (-1)
        (h:t) -> const (ord h) <$> writeIORef inp t
  export "putChar" putCh
  export "putInt"  putInt
  export "getChar" getCh
  let
    setupDemo mode name s = do
      Just b <- elemById $ name ++ "B"
      Just d <- elemById $ name ++ "Demo"
      Just r <- elemById $ mode ++ "Radio"
      void $ b `onEvent` Click $ const $ do
        setProp sEl "value" =<< getProp d "value"
        setProp r "checked" "true"
        setProp iEl "value" s
        setProp oEl "value" ""
  setupDemo "nat" "nat" ""
  setupDemo "lazyk" "lazyk" "gateman"
  setupDemo "fussyk" "fussyk" "(ignored)"
  setupDemo "crazyl" "crazyl" "length"
  setupDemo "crazyl" "rev" "stressed"
  setupDemo "crazyl" "sort" "froetf"
  void $ compB `onEvent` Click $ const $ do
    setProp skEl "value" ""
    setProp aEl "value" ""
    writeIORef bin []
    s <- getProp sEl "value"
    case parseProgram s of
      Left err -> setProp skEl "value" $ "error: " ++ show err
      Right sk -> do
        let
          f name = do
            Just el <- elemById $ name ++ "Radio"
            bool "" name . ("true" ==) <$> getProp el "checked"
        lang <- concat <$> mapM f ["nat", "lazyk", "fussyk", "crazyl"]
        let asm = compile (findLang lang) sk
        setProp skEl "value" $ showBabs sk
        setProp aEl "value" $ show asm
        writeIORef bin asm
  void $ runB `onEvent` Click $ const $ do
    setProp oEl "value" ""
    s <- getProp iEl "value"
    writeIORef inp s
    asm <- readIORef bin
    ffi "runWasmInts" asm :: IO ()

findLang :: String -> Lang
findLang "nat" = Nat
findLang "fussyk" = FussyK
findLang "crazyl" = CrazyL
findLang "lazyk" = LazyK
findLang _ = undefined
#endif
\end{code}

== Testing ==

We test our code with HUnit on
https://tromp.github.io/cl/lazy-k.html[known Lazy K examples]:

\begin{code}
#ifndef __HASTE__

mustParseProgram :: String -> Term
mustParseProgram = either (error "bad program") id . parseProgram

tests :: Test
tests = TestList
  [ "revK" ~: "diaper" ~?= runSim LazyK "repaid" rev
  , "revL" ~: "stressed" ~?= runSim CrazyL "desserts" "\\l.l(\\htx.t(\\cn.ch(xcn)))i(sk)"
  , "empty1" ~: "Hello, World!" ~?= runSim LazyK "Hello, World!" "\n"
  , "empty2" ~: "" ~?= runSim LazyK "" "\n"
  , "empty3" ~: "Hello, World!" ~?= runSim CrazyL "Hello, World!" "\n"
  , "kk256" ~: "" ~?= runSim LazyK "whatever" kk256
  , "5!" ~: "120" ~?= runSim Nat "" (unlines
    [ "Y=(\\z.zz)(\\z.\\f.f(zzf))"
    , "P=\\nfx.n(\\gh.h(gf))(\\u.x)(\\u.u)"
    , "M=\\mnf.m(nf)"
    , "z=\\n.n(\\x.sk)k"
    , "Y(\\fn.zn(\\fx.fx)(Mn(f(Pn))))(\\fx.f(f(f(f(fx)))))"
    ])
  , "primes" ~: let s = runSim FussyK "" pri in
    assertBool s $ "2 3 5 7 11 13" `isPrefixOf` s
  ]
  where
  kk256 = "k(k(sii(sii(sBi))))"
  runSim lang inp = sim lang inp . mustParseProgram
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

pri :: String
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
\end{code}

== Command-line UI ==

A REPL glues the above together. If no command-line arguments are given, then
we print bracket abstractions for each line of the program.

\begin{code}
main :: IO ()
main = do
  hSetBuffering stdout NoBuffering
  as <- getArgs
  let
    f lang = runInputT defaultSettings $ repl lang inArg []
    inArg = case as of
      (_:a:_) -> a
      _       -> ""
    repl lang inp = fix $ \rec env -> do
      getInputLine "> " >>= \case
        Nothing -> outputStrLn ""
        Just ln -> case parseEnv env ln of
          Left err -> outputStrLn err >> rec env
          Right env'@((s, t):_) -> do
            if s == "main" then do
              outputStrLn $ lang inp t
              rec env
            else do
              outputStrLn $ s ++ "=" ++ showBabs t
              rec env'
          _ -> error "unreachable"
  if null as then f $ const showBabs else case head as of
    "n"     -> f $ sim Nat
    "lazyk" -> f $ sim LazyK
    "k"     -> f $ sim FussyK
    "l"     -> f $ sim CrazyL
    "test"  -> void $ runTestTT tests
    "pri"   -> putStrLn $ take 70 $ sim FussyK "" $ mustParse pri
    "bm"    -> defaultMain $ pure $ bench "pri" $ whnf (\t -> "2 3 5 7 11 13" `isPrefixOf` sim LazyK t (mustParse pri)) ""
    "wasm"  -> print $ compile CrazyL $ mustParseProgram $ unlines
      [ "c=\\htcn.ch(tcn)"
      , "\\l.l(\\htx.t(chx))i(sk)"
      ]
    bad     -> putStrLn $ "bad command: " ++ bad
#endif
\end{code}
