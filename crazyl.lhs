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
V 65 (V 66 (V 256 (V 256 (...))))
------------------------------------------------------------------------------

where, as before, `Vxyz = zxy`, and the numbers are Church-encoded.

== Fussy K ==

Unfortunately, the reference implementation of Lazy K is sloppy with respect
to the output. Ideally, it should look for `V 256 x` in the
output list for any value of `x`, at which point the program should terminate,
but instead, the current item of the list is tested by applying it to the K
combinator, and if this returns 256 then the program halts. Indeed, the
documentation explicitly mentions that `K 256` is a valid end-of-output
marker.

However `V 256 x` behaves differently to `K 256`. For example
`V 256 x (KI) = x` while `K 256 (KI) = 256`. This complicates our
implementation.

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

We catch up with an old friend: an AST for lambda calculus terms. Once again,
we wish to eliminate all the lambda abstractions, leaving only variables and
applications.

\begin{code}
{-# LANGUAGE CPP #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE RecordWildCards #-}
#ifdef __HASTE__
{-# LANGUAGE OverloadedStrings #-}
import Haste.DOM
import Haste.Events
import Haste.Foreign
import Data.Bool
import Data.IORef
#else
{-# LANGUAGE TemplateHaskell #-}
import System.Console.Haskeline
import System.Environment
import System.IO
import Test.QuickCheck.All
import Criterion.Main hiding (env)
#endif
import Control.Monad
import Data.Char
import Data.Function (fix)
import Data.List
import qualified Data.Map as M
import Data.Map (Map, (!))
import Text.ParserCombinators.Parsec

infixl 5 :@
data Term = Var String | Term :@ Term | Lam String Term

instance Show Term where
  show (Var s)  = s
  show (l :@ r)  = show l ++ showR r where
    showR t@(_ :@ _) = "(" ++ show t ++ ")"
    showR t          = show t
  show _ = error "lambda present"
\end{code}

We define a few combinators. Some of their names are a nod to Smullyan's "To
Mock a Mockingbird".

\begin{code}
consBird, succBird, vireo, skk, vsk :: Term
consBird = mustParse "((bs)((b(bb))(ci)))"
succBird = Var "s" :@ Var "b"
vireo = Var "b" :@ Var "c" :@ (Var "c" :@ Var "i")
skk = Var "i"
vsk = vireo :@ Var "s" :@ Var "k"
\end{code}

Our parser closely follows https://tromp.github.io/cl/lazy-k.html[the grammar
specified in the description of Lazy K]. Differences:

 * We support lambda abstractions, e.g: `\x.x`
 * With `=`, we can assign terms to any letter except those in `"skiSKI"`, e.g:
 `c=\htcn.ch(tcn)`.
 * If left undefined, the letters `b` and `c` are interpreted as the B and C
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
      <|> Var . pure . toLower <$> oneOf "KS"
      <|> Var . pure <$> letter
      <|> between (char '(') (char ')') ccexpr
      <|> (char '`' >> (:@) <$> expr <*> expr)
      <|> (char '*' >> (:@) <$> iotaexpr <*> iotaexpr)
      <|> flip (foldr Lam) <$> between (char '\\' <|> char '\0955') (char '.')
        (many1 var) <*> ccexpr

  var = lookAhead (noneOf "skiSKI") >> pure <$> letter

  jotRev []       = skk
  jotRev ('0':js) = jotRev js :@ Var "s" :@ Var "k"
  jotRev ('1':js) = Var "s" :@ (Var "k" :@ jotRev js)
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
sub :: [(String, Term)] -> Term -> Term
sub env = \case
  x :@ y -> sub env x :@ sub env y
  Var s | Just t <- lookup s env -> t
        | otherwise              -> Var s
  Lam s t -> Lam s $ sub (filter ((/= s) . fst) env) t

parseProgram :: String -> Either String Term
parseProgram program = case go [] $ lines program of
  Left err -> Left err
  Right env -> maybe (Left "missing main") Right $ lookup "main" env
  where
  go acc [] = Right acc
  go acc (ln:rest) = case parseLine ln of
    Left e -> Left $ show e
    Right (s, t) -> go ((s, babs $ sub acc t):acc) rest
\end{code}

We can express SK terms in various languages as follows:

\begin{code}
dumpIota, dumpJot, dumpUnlambda :: Term -> String
dumpIota     = dumpWith '*' "*i*i*ii" "*i*i*i*ii"
dumpJot      = dumpWith '1' "11100"   "11111000"
dumpUnlambda = dumpWith '`' "k"       "s"

dumpWith :: Char -> String -> String -> Term -> String
dumpWith apCh kStr sStr = fix $ \f -> \case
  x :@ y  -> apCh:f x ++ f y
  Var "k" -> kStr
  Var "s" -> sStr
  _       -> error "SK terms only"
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
\end{code}

== Sick B ==

The bracket abstraction (i.e. removing lambdas) we use needs B and C
combinators (`(.)` and `flip` in Haskell) as well as the S and K combinators.
We also directly implement I combinators, rather than making do with SKK.

A straightforward recursion computes the De Bruijn indices.
Because we may be asked to perform bracket abstraction on the output of our
bracket abstraction routine, we must support B and C when converting to De
Bruijn indices.

\begin{code}
toDeb :: Deb repr => [String] -> Term -> repr
toDeb env = \case
  Var s -> case elemIndex s env of
    Nothing -> case s of
      "s" -> lam $ lam $ lam $ su(su ze) # ze # (su ze # ze)
      "b" -> lam $ lam $ lam $ su(su ze)      # (su ze # ze)
      "c" -> lam $ lam $ lam $ su(su ze) # ze #  su ze
      "k" -> lam $ lam $ su ze
      "i" -> lam ze
      _ -> error $ s <> " is free"
    Just n -> iterate su ze !! n
  Lam s t -> lam $ toDeb (s:env) t
  x :@ y -> toDeb env x # toDeb env y
\end{code}

Now we can apply http://okmij.org/ftp/tagless-final/ski.pdf[a powerful bracket
abstraction algorithm due to Oleg Kiselyov].

Again we use a tagless final representation for the output of the algorithm.

\begin{code}
infixl 5 ##
class SickB repr where
  kS :: repr
  kI :: repr
  kC :: repr
  kK :: repr
  kB :: repr
  (##) :: repr -> repr -> repr
\end{code}

We declare an instance so we can convert back to our `Term` AST. This allows
messy shortcuts suitable for prototyping: we can print `Term` values, we can
reuse code from the previous compiler, and we can add new combinators without
adding them to the typeclass since a `Var` holds any string.

\begin{code}
instance SickB Term where
  kS = Var "s"
  kI = Var "i"
  kC = Var "c"
  kK = Var "k"
  kB = Var "b"
  e1 ## e2 = e1 :@ e2
\end{code}

We introduce another data type because the algorithm needs to distinguish
between closed terms, and several kinds of unclosed terms. We return a closed
term, but along the way we manipulate open terms.

\begin{code}
data Oleg repr = C {unC :: repr} | N (Oleg repr) | W (Oleg repr) | V
instance SickB repr => Deb (Oleg repr) where
  ze = V
  su = W
  l # r = case (l, r) of
    (W e, V) -> N e
    (V, W e) -> N $ C (kC ## kI) # e
    (N e, V) -> N $ C kS # e # C kI
    (V, N e) -> N $ C (kS ## kI) # e
    (C d, V) -> N $ C d
    (V, C d) -> N $ C $ kC ## kI ## d
    (V, V)   -> N $ C $ kS ## kI ## kI

    (W e1, W e2) -> W $ e1 # e2
    (W e, C d)   -> W $ e # C d
    (C d, W e)   -> W $ C d # e
    (W e1, N e2) -> N $ C kB # e1 # e2
    (N e1, W e2) -> N $ C kC # e1 # e2
    (N e1, N e2) -> N $ C kS # e1 # e2
    (C d, N e)   -> N $ C (kB ## d) # e
    (N e, C d)   -> N $ C (kC ## kC ## d) # e
    (C d1, C d2) -> C $ d1 ## d2
  lam = \case
    V   -> C kI
    C d -> C $ kK ## d
    N e -> e
    W e -> C kK # e

babs :: Term -> Term
babs = unC . toDeb []
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
combs = "sickb0+<>."
\end{code}

The heap is organized as an array of 8-byte entries, each consisting of two
4-byte combinators `x` and `y`. The meaning of such an entry is `xy`.

A negative 4-byte value represents one of the primitive combinators.
Otherwise it is the address of another 8-byte entry in the heap.

This encoding scheme means if a term consists of a single primitive combinator,
such as K, then we must represent it as IK since at minimum a cell holds two
combinators.

\begin{code}
wlen :: [a] -> Int
wlen = (4*) . length

enCom :: String -> Int
enCom [c] | Just n <- elemIndex c combs = -n - 1
enCom s = error $ show s

encAt :: Int -> Term -> [Int]
encAt _ (Var a :@ Var b) = [enCom a, enCom b]
encAt n (Var a :@ y)     = enCom a : n + 8 : encAt (n + 8) y
encAt n (x     :@ Var b) = n + 8 : enCom b : encAt (n + 8) x
encAt n (x     :@ y)     = n + 8 : nl : l ++ encAt nl y
  where l  = encAt (n + 8) x
        nl = n + 8 + wlen l
encAt _ _                = error "want application"

dump :: VM -> String
dump VM{..} = unlines $ take 50 . f <$> ps where
  f a | a < 0 = pure $ combs!!(-a - 1)
  f a         = "(" ++ f (de a) ++ f (de $ a + 4) ++ ")"
  ps = ip:[de $ 4 + de p | p <- [sp, sp + 4..maxSP - 4]]
  de k | Just v <- M.lookup k mem = v
       | otherwise = error $ "bad deref: " ++ show k
\end{code}

We place the Church encoded integers from [0..256] in linear memory starting
from 0.  Each takes one 8-byte cell, so that the Church encoding of 'n' lies at
address '8n' in memory.

Zero is represented by SK, and 'n + 1' is represented by 'm n' where 'm' is
the combinator that computes the successor of a Church number. We place the
definition of 'm' just after the Church numbers, that is, at memory address
`8*257`.

\begin{code}
gen :: [Int]
gen = enCom "s" : enCom "k" :           -- Zero
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
  LazyK   -> t :@ (Var "<" :@ vireo) :@ Var ">" :@ Var "+" :@ Var "0"
  FussyK  -> t :@ (Var "<" :@ vireo) :@ Var ">"
  CrazyL  -> t :@ (Var "<" :@ consBird) :@ Var ">" :@ Var "."
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
  's' -> rec $ upd hp (hp + 8) . pop 2 . putHP [arg 0, arg 2, arg 1, arg 2, hp, hp + 8]
  'i' -> rec $ setIP (arg 0) . pop 1
  'c' -> rec $ putHP [arg 0, arg 2] . upd hp (arg 1) . pop 2
  -- Unmemoized: 'k' -> rec $ setIP (arg 0) . pop 2
  'k' -> rec $ upd (enCom "i") (arg 0) . pop 1
  'b' -> rec $ putHP [arg 1, arg 2] . upd (arg 0) hp . pop 2
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

For Lazy K and Fussy K, we choose `x` to be the V combinator and
we replace the entry with `Vn(<V)`, where `n` is the Church encoding of the
next byte of input, or 256 if there is no more input. Recall we have placed
[0..256] at the beginning of memory, so this is just the address `8*n`.

For Crazy L, we choose `x` to be the cons combinator for right-fold
representations of lists, and we replace the entry with `xn(<0)` where `n` is
the Church encoding of the next byte of input, or `SK` if there is no more
input.

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
    LazyK   -> if ax == 256 then "" else
      chr ax : rec (upd (hp + 8) (enCom "0") . putHP
        [arg 0, enCom ">", hp, enCom "+"] . setAX 0)
  -- I combinator with side effect.
  '+' -> rec $ setIP (arg 0) . pop 1 . setAX (ax + 1)
  '>' -> rec $ upd hp (hp + 8) . pop 1 . putHP
    [arg 0, enCom "+", enCom "0", arg 1]
  '.' -> ""
  -- Lazy input. If we reach here, then IP == [[SP]].
  '<' | CrazyL <- lang -> case input of
     (h:t) -> exec $ putHP [arg 0, ord h * 8, enCom "<", arg 0] $ upd hp (hp + 8) vm { input = t }
     _     -> rec $ upd (enCom "s") (enCom "k")
  '<' -> exec
    $ putHP [arg 0, (case input of { (h:_) -> ord h; _ -> 256 }) * 8, enCom "<", arg 0]
    $ upd hp (hp + 8)
    $ vm { input = case input of { (_:t) -> t; _ -> [] }}
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

  * We give output bytes to `f`.
  * We call `g` to get the next input byte. This function should return 256
    if there is no more input.
  * The Nat language calls `h` to return an integer.

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
  i32lt_u  = 0x49
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
    'k' -> updatePop 1 (asmCom "i") (asmArg 0) ++ loop
    's' -> withHeap (asmArg <$> [0, 2, 1, 2]) (updatePop 2 (hNew 0) (hNew 1)) ++ loop
    '>' -> withHeap [asmArg 0, asmCom "+", asmCom "0", asmArg 1] (updatePop 1 (hNew 0) (hNew 1)) ++ loop
    '.' -> [br, exitLabel]  -- br exit
    'i' -> concat [asmIP $ asmArg 0, asmPop 1, loop]
    '<' | CrazyL <- mode -> concat
      [ [0x10, 1, teelocal, ip]  -- Get next character in IP.
      , [i32const, 128, 2, i32lt_u, 4, 0x40]  -- if < 256
      , withHeap [asmArg 0, [getlocal, ip, i32const, 8, i32mul], asmCom "<", asmArg 0] (updatePop 0 (hNew 0) (hNew 1))
      , [5]  -- else
      , updatePop 0 (asmCom "s") (asmCom "k")
      , [0xb]  -- end if
      , loop
      ]
    '<' -> withHeap [asmArg 0, [0x10, 1, i32const, 8, i32mul], asmCom "<", asmArg 0] (updatePop 0 (hNew 0) (hNew 1)) ++ loop
    'b' -> withHeap [asmArg 1, asmArg 2] (updatePop 2 (asmArg 0) (hNew 0)) ++ loop
    'c' -> withHeap [asmArg 0, asmArg 2] (updatePop 2 (hNew 0) (asmArg 1)) ++ loop
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
        [] -> do
          writeIORef inp []
          pure 256
        (h:t) -> do
          writeIORef inp t
          pure $ ord h
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
        setProp skEl "value" $ show sk
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

We test our code with QuickCheck on
https://tromp.github.io/cl/lazy-k.html[known Lazy K examples]:

\begin{code}
#ifndef __HASTE__
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

kk256 = "k(k(sii(sii(sbi))))"

prop_rev s = sim LazyK t (mustParse rev)   == reverse t where t = take 10 s
prop_id s  = sim LazyK s (mustParse "")    == s
prop_emp s = sim LazyK s (mustParse kk256) == ""
prop_pri   = "2 3 5 7 11 13" `isPrefixOf` sim LazyK  "" (mustParse pri)
prop_pri'  = "2 3 5 7 11 13" `isPrefixOf` sim FussyK "" (mustParse pri)

mustParseProgram :: String -> Term
mustParseProgram = either undefined id . parseProgram

prop_fac5 = ("120" ==) $ sim Nat "" $ mustParseProgram $ unlines
  --"Y=ssk(s(k(ss(s(ssk))))k)",
  [ "Y=(\\z.zz)(\\z.\\f.f(zzf))"
  , "P=\\nfx.n(\\gh.h(gf))(\\u.x)(\\u.u)"
  , "M=\\mnf.m(nf)"
  , "z=\\n.n(\\x.sk)k"
  , "Y(\\fn.zn(\\fx.fx)(Mn(f(Pn))))(\\fx.f(f(f(f(fx)))))"
  ]

return []
runAllTests = $quickCheckAll
\end{code}

== Command-line UI ==

A REPL glues the above together. The first two command-line arguments determine
the language (or dump format) and the input to the program; if omitted, they
default to Crazy L and the empty string.

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
    repl lang inp env = do
      let rec = repl lang inp
      getInputLine "> " >>= \case
        Nothing -> outputStrLn ""
        Just ln -> case parseLine ln of
          Left err  -> do
            outputStrLn $ "parse: " ++ show err
            rec env
          Right (s, rhs) -> do
            let t = babs $ sub env rhs
            if s == "main" then do
              outputStrLn $ lang inp t
              rec env
            else do
              outputStrLn $ s ++ "=" ++ show t
              rec ((s, t):env)

  if null as then f $ sim CrazyL else case head as of
    "n"     -> f $ sim Nat
    "lazyk" -> f $ sim LazyK
    "k"     -> f $ sim FussyK
    "l"     -> f $ sim CrazyL

    "sk"    -> f $ const show
    "iota"  -> f $ const dumpIota
    "jot"   -> f $ const dumpJot
    "unl"   -> f $ const dumpUnlambda
    "test"  -> void runAllTests

    "rev"   -> putStrLn $ sim LazyK "0123456789abcdef" $ mustParse rev
    "pri"   -> putStrLn $ take 20 $ sim FussyK "" $ mustParse pri
    "bm"    -> defaultMain $ pure $ bench "rev" $ whnf (\t -> sim LazyK t (mustParse rev) == reverse t) "0123456789abcdef"
    "wasm"  -> print $ compile CrazyL $ mustParseProgram $ unlines
      [ "c=\\htcn.ch(tcn)"  -- cons
      , "\\l.l(\\htx.t(chx))i(sk)"
      ]
    bad     -> putStrLn $ "bad command: " ++ bad
#endif
\end{code}
