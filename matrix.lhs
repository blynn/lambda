= Matrix Combinators =

[pass]
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
<p>Lambda term: <textarea id="input" rows="1" cols="40">\a b c d.d c b a</textarea></p>

<button id="matrix">Matrix</button>
<button id="matrixOpt">Matrix Optimized</button>
<p><textarea id="output" rows="3" cols="80" readonly></textarea></p>

<script>
"use strict";
function hideshow(s) {
  const x = document.getElementById(s);
  x.style.display = x.style.display === "none" ? "block" : "none";
}
let getcharBuf, putcharBuf;

function setup(instance) {
  function go(s, inp, out) {
    document.getElementById(s).onclick = function() {
      getcharBuf = document.getElementById(inp).value;
      document.getElementById(out).value = "";
      putcharBuf = "";
      instance.exports[s + "Main"]();
      document.getElementById(out).value = putcharBuf;
    };
  }
  go("matrix", "input", "output");
  go("matrixOpt", "input", "output");
}

WebAssembly.instantiateStreaming(fetch('matrix.wasm'), { env:
  { getchar: () => {
      if (!getcharBuf.length) throw "eof";
      const n = getcharBuf.charCodeAt(0);
      getcharBuf = getcharBuf.slice(1)
      return n;
    }
  , putchar: c => putcharBuf += String.fromCharCode(c)
  , eof: () => getcharBuf.length == 0
  }}).then(obj => setup(obj.instance));
</script>

<p><a onclick='hideshow("stuff");'>&#9654; Toggle reused code</a></p>
<div id='stuff' style='display:none'>
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

\begin{code}
{- For GHC, uncomment the following, and comment the next block.
{-# LANGUAGE LambdaCase, BlockArguments #-}
import Control.Applicative
import Control.Arrow
-}

module Main where
import Base
import System
demo f = interact $ either id (show . snd . f . deBruijn) . parseLC
foreign export ccall "matrixMain" matrixMain
matrixMain = demo matrix
foreign export ccall "matrixOptMain" matrixOptMain
matrixOptMain = demo matrixOpt

data Charser a = Charser { getCharser :: String -> Either String (a, String) }
instance Functor Charser where fmap f (Charser x) = Charser $ fmap (first f) . x
instance Applicative Charser where
  pure a = Charser \s -> Right (a, s)
  f <*> x = Charser \s -> do
    (fun, t) <- getCharser f s
    (arg, u) <- getCharser x t
    pure (fun arg, u)
instance Monad Charser where
  Charser f >>= g = Charser $ (good =<<) . f
    where good (r, t) = getCharser (g r) t
  return = pure
instance Alternative Charser where
  empty = Charser \_ -> Left ""
  (<|>) x y = Charser \s -> either (const $ getCharser y s) Right $ getCharser x s

sat f = Charser \case
  h:t | f h -> Right (h, t)
  _ -> Left "unsat"

eof = Charser \case
  [] -> Right ((), "")
  _ -> Left "want EOF"

char :: Char -> Charser Char
char = sat . (==)

string :: String -> Charser String
string s = mapM char s

isDigit c = '0' <= c && c <= '9'
isAlphaNum c
  | 'A' <= c && c <= 'Z' = True
  | 'a' <= c && c <= 'z' = True
  | isDigit c = True
  | otherwise = False

lcTerm = lam <|> app where
  lam = flip (foldr Lam) <$> (lam0 *> some var <* lam1) <*> lcTerm
  lam0 = str "\\" <|> str "\955"
  lam1 = str "->" <|> str "."
  app = foldl1 App <$> some (Var <$> var <|> str "(" *> lcTerm <* str ")")
  var = some (sat isAlphaNum) <* whitespace

str = (<* whitespace) . string
whitespace = many $ sat $ (`elem` " \n")

parseLC = fmap fst . getCharser (whitespace *> lcTerm <* eof)

parseBulk = fmap fst . getCharser do
  whitespace
  c <- sat (`elem` "BCS")
  whitespace
  n <- foldl (\n c -> n*10 + fromEnum c - fromEnum '0') 0 <$> some (sat isDigit)
  eof
  pure ([c], n)

instance Show Term where
  show (Lam s t)  = "\955" ++ s ++ showB t where
    showB (Lam x y) = " " ++ x ++ showB y
    showB expr      = "." ++ show expr
  show (Var s)    = s
  show (App x y)  = showL x ++ showR y where
    showL (Lam _ _) = "(" ++ show x ++ ")"
    showL _         = show x
    showR (Var s)   = ' ':s
    showR _         = "(" ++ show y ++ ")"

data Term = Var String | App Term Term | Lam String Term
data Peano = S Peano | Z
data DB = N Peano | L DB | A DB DB | Free String

index x xs = lookup x $ zip xs $ iterate S Z

deBruijn = go [] where
  go binds = \case
    Var x -> maybe (Free x) N $ index x binds
    Lam x t -> L $ go (x:binds) t
    App t u -> A (go binds t) (go binds u)
\end{code}
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
</div>
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Kiselyov describes a linear-time algorithm to convert a lambda term to
combinators, provided the input is written with De Bruijn indexes in unary,
and provided we allow bulk combinators.

Can we do better with different bulk combinators? In particular, we would like
to drop the unary representation proviso.

This is related to
link:../compiler/golf.html[my thoughts on combinator golf], and again, a
trivial answer exists. We define a complex combinator $X$ which takes a
representation of a lambda term and returns an equivalent combinator term. In
other words, this single combinator implies an optimal algorithm for
converting lambda terms to combinators!

We would like a more honest solution, though we are perhaps willing to tolerate
more complexity than most.

Inspired by Kiselyov's method of keeping track of the variables used on either
side of an application ("directing whole environments rather than single
variables"), we define bulk combinators that correspond to matrices of bits as
follows.

Let $X$ be a $m \times n$ matrix of bits, and let $b_{ij}$ be the bit in the
$i$th row and $j$ column.

An empty matrix represents the identity combinator $I$.
Otherwise, $X$ represents the combinator:

\[
\lambda f_1 ... f_m x_1 ... x_n . (f_1 (b_{11} ? x_1) ... (b_{1n} ? x_n)) ... (f_m (b_{m1} ? x_1) ... (b_{mn} ? x_n))
\]

where $b ? x$ means $x$ if $b = 1$, and is omitted if $b = 0$. We mostly
stick to the $m\le 2$ cases.

Examples:

  * $[0] a b = a$: the $K$ combinator.

  * $[00010101] a b c d e f g h i = a e g i$. We see that we can immediately write down the matrix combinator that selects any given subset of arguments.

  * $[0, 1] a b c = a (b c)$: the $B$ combinator.

  * $[1, 0] a b c = (a c) b$: the $C$ combinator.

  * $[1, 1] a b c = (a c) (b c)$: the $S$ combinator.

  * $[0, 0] a b c = a b$: the $BK$ combinator.

  * $([11,00]I) a b c = (I b c) a = b c a$: the $R$ combinator.

These matrix combinators suggest a K-optimized bracket abstraction algorithm in
the style of Kiselyov, but with more efficient projection: we combine
consecutive lambdas as if we were building link:bohm.html[a Böhm node], and use
a one-row matrix (or vector) to describe a combinator that selects out exactly
the arguments that are needed. If all arguments are needed, then we skip this
selector vector.

\begin{code}
data CL = Com [[Bool]] | ComFree String | CL :@ CL
instance Show CL where
  showsPrec p = \case
    Com xs -> case xs of
      [] -> ('I':)
      _ -> ('[':) . foldr (.) (']':) (intersperse (',':) $ foldr (.) id . map (shows . fromEnum) <$> xs)
    ComFree s -> (if p > 0 then (' ':) else id) . (s++)
    t :@ u -> showParen (p > 0) $ showsPrec 0 t . showsPrec 1 u

matrix = \case
  N Z -> ([True], Com [])
  N (S e) -> first (False:) $ matrix $ N e
  L e -> sledL 1 e
  A e1 e2 -> matrix e1 ## matrix e2
  Free s -> ([], ComFree s)
  where
  sledL n = \case
    L e -> sledL (n + 1) e
    e -> let
      (g, d) = matrix e
      present = reverse $ take n (g ++ repeat False)
      in (if and present then id else (([], Com [present]) ##)) (drop n g, d)

  ([], d1) ## ([], d2) = ([], d1 :@ d2)
  (g1, d1) ## (g2, d2) = (g, Com [p1, p2] :@ d1 :@ d2)
    where
    zs = zipWithDefault False (,) g1 g2
    g = uncurry (||) <$> zs
    (p1, p2) = unzip $ reverse $ filter (uncurry (||)) zs

zipWithDefault d f     []     ys = f d <$> ys
zipWithDefault d f     xs     [] = flip f d <$> xs
zipWithDefault d f (x:xt) (y:yt) = f x y : zipWithDefault d f xt yt
\end{code}

Instead of handling one application at a time, we could handle an entire Böhm
node in one step, generating a matrix with $m$ rows for a Böhm node with $m$
children. This may result in savings. For example, the term $\lambda a b c d .
a (b d) (c d)$ could be written $[0, 1, 1]$ rather than $[1101,0011] [0,1]$.

A few more lines give us eta-optimization. We generalize $BIx \rightarrow x$
by looking for $[0 ... 0] I x$. A recursive helper `etaRight` handles the
$BxI \rightarrow x$ case.

We also apply the optimizations:

  * $[10x, 01y] I I \rightarrow [x, y]$
  * $[1 ... 10, 0 ... 01] I I \rightarrow I$

\begin{code}
matrixOpt = \case
  N Z -> ([True], Com [])
  N (S e) -> first (False:) $ matrixOpt $ N e
  L e -> sledL 1 e
  A e1 e2 -> matrixOpt e1 ## matrixOpt e2
  Free s -> ([], ComFree s)
  where
  sledL n = \case
    L e -> sledL (n + 1) e
    e -> let
      (g, d) = matrixOpt e
      present = reverse $ take n (g ++ repeat False)
      in (if and present then id else (([], Com [present]) ##)) (drop n g, d)

([], d1) ## ([], d2) = ([], d1 :@ d2)
(g1, d1) ## (g2, d2)
  | Com [] <- d1, Com [] <- d2 = \cases
    | not $ or $ last p1 : init p2 -> go $ Com []
    | True:False:t1 <- p1, False:True:t2 <- p2 -> go $ Com [t1, t2]
    | otherwise -> common
  | Com [] <- d1, not $ or p1 = go d2
  | otherwise = common
  where
  x = Com [p1, p2]
  common = go $ x :@ d1 :@ d2
  zs = zipWithDefault False (,) g1 g2
  go = (uncurry (||) <$> zs,) . etaRight
  (p1, p2) = unzip $ reverse $ filter (uncurry (||)) zs
  etaRight (Com [False:t1, True:t2] :@ d :@ Com []) = etaRight $ Com [t1, t2] :@ d
  etaRight d = d
\end{code}

== Complexity ==

A lambda term with $n$ distinct variables requires $O(n\log n)$ bits to
represent. Thus there are some terms that are more efficient with our scheme:
in particular, any one- or two-row matrix combinator taking $n$ arguments only
needs $O(n)$ bits.

My guess is the worst case is the classic $\lambda x_1 ... x_n.x_n ... x_1$,
which only requires $O(n\log n)$ bits as a lambda term, but $O(n^2)$ bits
with matrix combinators.

When variables are written in unary, our algorithm is linear, and compares
favourably with Kiselyov's algorithm because we've chosen more complex bulk
combinators.

== Two Rows ==

It could be worth restricting our attention to $2 \times n$ matrices. The empty
$2 \times 0$ matrix is the identity combinator. Any one-row matrix $[r]$ is
equivalent to $[0 ... 0,r] I$.

We could then design a virtual machine that only needs to handle two-row matrix
combinators. It ought to be easy to implement as there is only kind of
combinator. At the same time, we only have one (bulk) combinator per
application or lambda streak.

We can impose a maximum number of columns per combinator by observing:

\[
[r] = [0 ... 0,r_1] ([0 ... 0,r_2] I)
\]

if $r = r_1 \diamond r_2$, where $(\diamond)$ denotes concatenation; and:

\[
[x,y] = [0,1] [x_1,y_1] ([0 ... 0,1 ... 1] [x_2,y_2])
\]

where $x = x_1 \diamond x_2, y = y_1 \diamond y_2$ and where each $x_i, y_i$
have the same length.
