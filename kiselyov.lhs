= Kiselyov Combinator Translation =

[pass]
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
<p>Lambda term: <textarea id="input" rows="1" cols="40">\a b c d.d c b a</textarea></p>

<button id="plain">Plain</button>
<button id="optimizeK">K-optimized</button>
<button id="optimizeEta">Eta-optimized</button>
<br>
<button id="rawBulk">Bulk</button>
<button id="optimizeBulk">Bulk, Optimized</button>
<button id="linBulk">Bulk, Linear Expand</button>
<button id="logBulk">Bulk, Log Expand</button>
<br>
<br>
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

  go("plain", "input", "output");
  go("optimizeK", "input", "output");
  go("optimizeEta", "input", "output");
  go("rawBulk", "input", "output");
  go("linBulk", "input", "output");
  go("logBulk", "input", "output");

  go("optimizeBulk", "input", "output");

  go("linearbb", "inputbb", "outputbb");
  go("logbb", "inputbb", "outputbb");
}

WebAssembly.instantiateStreaming(fetch('kiselyov.wasm'), { env:
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
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
<p><a onclick='hideshow("stuff");'>&#9654; Toggle boilerplate, parser</a></p>
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

foreign export ccall "plainMain" plainMain
plainMain = demo plain
foreign export ccall "optimizeKMain" optimizeKMain
optimizeKMain = demo optK
foreign export ccall "optimizeEtaMain" optimizeEtaMain
optimizeEtaMain = demo optEta
foreign export ccall "rawBulkMain" rawBulkMain
rawBulkMain = demo $ bulkPlain bulk
foreign export ccall "optimizeBulkMain" optimizeBulkMain
optimizeBulkMain = demo $ bulkOpt bulk
foreign export ccall "linBulkMain" linBulkMain
linBulkMain = demo $ bulkOpt breakBulkLinear
foreign export ccall "logBulkMain" logBulkMain
logBulkMain = demo $ bulkOpt breakBulkLog
foreign export ccall "linearbbMain" linearbbMain
linearbbMain = interact $ either id (show . uncurry breakBulkLinear) . parseBulk
foreign export ccall "logbbMain" logbbMain
logbbMain = interact $ either id (show . uncurry breakBulkLog) . parseBulk

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
\end{code}

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
</div>
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

http://okmij.org/ftp/tagless-final/ski.pdf[Oleg Kiselyov's algorithm for
compiling lambda calculus to combinators] seems unreasonably effective in
practice.

== A Special Bracket Abstraction ==

The heart of the algorithm is a specialized bracket abstraction. Let $d_1$ be a
combinator and 'eta-expand' it any number of times, that is, nest it in a bunch
of lambda abstractions and apply to each of the new variables, starting from
the variable bound in the outermost lambda. For example: eta-expanding $d_1$
three times yields:

\[
\lambda \lambda \lambda d_1 2 1 0
\]

when we write it with link:cl.html[De Bruijn indices],
Let $d_2$ be a combinator, and eta-expand it any number of times. For example,
eta-expanding $d_2$ twice yields:

\[
\lambda \lambda d_2 1 0
\]

Let $d$ be the result of applying the innermost body of one to that of the
other, while sharing the same variables:

\[
d = \lambda \lambda \lambda d_1 2 1 0 (d_2 1 0)
\]

How do you eliminate the added variables? That is, can you write $d$ just using
$d_1, d_2$ and some fixed set of combinators?

We'll have more to say about eta-expansion in link:bohm.html[our tour of Böhm's
Theorem], but we do want to mention a subtlety here. The eta-expansion of a
lambda term $F$ is defined to be $\lambda F 0$. Eta-expanding again results in
$\lambda (\lambda F 0) 0$, which reduces to $\lambda F 0$, getting us nowhere.
On the other hand, eta-expanding just the body of the term, namely $F 0$,
results in the less trivial $\lambda \lambda F 1 0$.

So when we say we eta-expand a term multiple times, we mean we expand the part
after the lambdas (the body of a Böhm tree node), and not the entire term.

== Solution ==

We choose the combinators $B, R, S$:

\[
\begin{align}
  Babc & = a(bc) \\
  Rabc & = bca \\
  Sabc & = ac(bc)
\end{align}
\]

(Schönfinkel would have written $Z, TT, S$; in Haskell, these are `(.)`, `flip
flip`, and the Reader instance of `ap` or `(<*>)`.)

Then the following solves our example puzzle:

\[
d = R d_2 (BS(B(BS)d_1))
\]

What's the secret to deriving this? We could run a standard bracket abstraction
algorithm a few times, but better to take advantage of the eta-expansions.

First, some notation. Let $n\models d$ denote the term $d$ eta-expanded $n$
times. For example, $3 \models d = \lambda \lambda \lambda d 2 1 0$.

Then we may restate our problem as follows. Given $n_1 \models d_1$ and $n_2
\models d_2$, our goal is find a combinator term equivalent to:

\[
d = \lambda ... \lambda d_1 (n_1 - 1) ... 0 (d_2 (n_2 - 1) ... 0)
\]

where there are $\max(n_1, n_2)$ lambda abstractions. (This equation would be
more compact if we started our De Bruijn indices from 1, but I refuse to budge
from 0-indexing!)

The trick is to use induction. Define the operation $\amalg$ by:

\[
\begin{align}
  0 \models d_1 \amalg 0 \models d_2 &=& d_1 d_2 \\
  n_1 + 1 \models d_1 \amalg 0 \models d_2 &=& 0 \models R d_2 \amalg n_1 \models d_1 \\
  0 \models d_1 \amalg n_2 + 1 \models d_2 &=& 0 \models B d_1 \amalg n_2 \models d_2 \\
  n_1 + 1 \models d_1 \amalg n_2 + 1 \models d_2 &=& n_1 \models (0 \models S \amalg n_1 \models d_1) \amalg n_2 \models d_2
\end{align}
\]

We can mechanically verify $d = n_1 \models d_1 \amalg n_2 \models d_2$ is a
solution.

== Combinators ==

Adding a little code turns the above bracket abstraction for variable-sharing
eta-expansions into an algorithm for translating any lambda calculus term to
combinators.

Some standard combinators:

\[
\begin{align}
  Ia & = a \\
  Kab & = a \\
  Cabc & = acb
\end{align}
\]

(Schönfinkel would have written $I, C, T$; in Haskell, these are `id`, `const`,
and `flip`.)

Now, given a lambda term, we rewrite it in De Bruijn form, where each index is
written using a unary representation known as the Peano numbers:

\[
z, sz, ssz, sssz, ...
\]

For example, the standard combinators we just mentioned are:

\[
\begin{align}
  I & = \lambda z \\
  K & = \lambda \lambda sz \\
  C & = \lambda \lambda \lambda (ssz)z(sz)
\end{align}
\]

We rewrite our De Bruijn conversion code to use Peano numbers:

\begin{code}
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

Define:

\[
\newcommand{\ceil}[1]{\lceil #1 \rceil}
\begin{array}{c c l}
\ceil{z} &=& 1 \models I \\
\ceil{s e} &=& n + 1 \models (0 \models K \amalg n \models d) \\
\ceil{\lambda e} &=&
  \left\{ 
    \begin{array}{ c l }
      0 \models K d & \quad \textrm{if } n = 0 \\
      n - 1 \models d & \quad \textrm{otherwise}
    \end{array}
  \right. \\
\ceil{e_1 e_2} &=& \max(n_1, n_2) \models (n_1 \models d_1 \amalg n_2 \models d_2)
\end{array}
\]

where we have recursively computed:

\[
\begin{align}
n \models d = \ceil{e} \\
n_1 \models d_1 = \ceil{e_1} \\
n_2 \models d_2 = \ceil{e_2}
\end{align}
\]

If $e$ is a closed lambda term and $n \models d = \ceil{e}$, then $n = 0$ and
the combinator $d$ is the bracket abstraction of $e$. More generally, for any
(possibly open) lambda term $e$, adding $n$ abstractions to $e$ will close it,
and $d$ is the bracket abstraction of the result.

We'll wave our hands to explain the algorithm and hope informality helps us
get away with less theory. See Kiselyov for details.

The first rule says $z$ translates to $1\models I$, namely a thing that becomes
the $I$ combinator after adding one lambda. Indeed, if we apply the third rule,
we find $\ceil{\lambda Z} = 0\models I$, as expected.

For the second rule, consider the example:

\[
\begin{align}
\ceil{s z} &=& 2 \models (0 \models K \amalg 1 \models I) \\
&=& 2 \models BKI
\end{align}
\]

As expected, we need 2 lambdas to close the term $sz$, and roughly speaking,
the $K$ combinator shifts the focus from the second variable to the first. This
generalizes to higher De Bruijn indices.

The third rule has two cases. If the $d$ is already a closed term, then the $K$
combinator ignores the new variable and simply returns $d$. Otherwise, we
decrement the count indicating how many more lambdas we need to close the term.

The last rule uses the specialized bracket abstraction, which captures the idea
of applying a possibly open term to another. We see why we want to share
variables: when we eventually close the term with lambda abstractions, the
indices in both sides of the application refer to the same set of variables.

In our code, the tuple `(n, d)` represents $n \models d$.

\begin{code}
data CL = Com String | CL :@ CL
instance Show CL where
  showsPrec p = \case
    Com s -> (if p > 0 then (' ':) else id) . (s++)
    t :@ u -> showParen (p > 0) $ showsPrec 0 t . showsPrec 1 u

convert (#) = \case
  N Z -> (1, Com "I")
  N (S e) -> (n + 1, (0, Com "K") # t) where t@(n, _) = rec $ N e
  L e -> case rec e of
    (0, d) -> (0, Com "K" :@ d)
    (n, d) -> (n - 1, d)
  A e1 e2 -> (max n1 n2, t1 # t2) where
    t1@(n1, _) = rec e1
    t2@(n2, _) = rec e2
  Free s -> (0, Com s)
  where rec = convert (#)

plain = convert (#) where
  (0 , d1) # (0 , d2) = d1 :@ d2
  (0 , d1) # (n , d2) = (0, Com "B" :@ d1) # (n - 1, d2)
  (n , d1) # (0 , d2) = (0, Com "R" :@ d2) # (n - 1, d1)
  (n1, d1) # (n2, d2) = (n1 - 1, (0, Com "S") # (n1 - 1, d1)) # (n2 - 1, d2)
\end{code}

I have a poor intuitive grasp why Kiselyov's algorithm seems to beat
traditional methods, but it's clear the algorithm sticks out from the rest of
the pack. Roughly speaking, all of them start from the innermost lambdas and
work their way outwards, but while traditional algorithms treat locally unbound
variables as opaque constants, Kiselyov's algorithm meticulously records
information about each unbound variable: the left of a $\models$ is the number
of lambdas needed to close the variable on the right. Perhaps this
industriousness leads to better performance.

== Lazy Weakening ==

Consider the De Bruijn lambda term:

\[
\lambda \lambda (sz) (sz)
\]

The second variable is unused, so we could derive the following combinator:

\[
K(\lambda z z) = K(SII)
\]

However, recall:

\[
\ceil{sz} = 2 \models BKI
\]

which leads to the verbose:

\[
\ceil{\lambda \lambda (sz) (sz)} = 0 \models S(BS(BKI))(BKI)
\]

A better strategy is to refrain from immediately converting terms like $sz$ to
combinators. Instead, on encountering $s$, we leave a sort of IOU that we
eventually redeem with a combinator. If both sides of an application turn out
to ignore the same variable, we merge the IOUs into one, resulting in savings.

To this end, we replace the number $n$ in $n \models d$ with a list of $n$
booleans, which we often denote by $\Gamma$. We use cons lists, and denote cons
by $(:)$, and the empty list by $\emptyset$.

The list item at index $k$ is `True` when the variable with De Bruijn index $k$
appears at least once in the term, in which case we have already generated
combinators to get at it. Otherwise it is unused and marked `False`, and is
only cashed in for a $K$ combinator when eventually lambda-abstracted,
provided it has never been upgraded to `True` because the other branch of an
application uses the corresponding variable.

\[
\begin{align}
  \emptyset \models d_1 \amalg \emptyset \models d_2 &=& d_1 d_2 \\
  \emptyset \models d_1 \amalg T : \Gamma \models d_2 &=& \emptyset \models B d_1 \amalg \Gamma \models d_2 \\
  T : \Gamma \models d_1 \amalg \emptyset \models d_2 &=& \emptyset \models R d_2 \amalg \Gamma \models d_1 \\
  T : \Gamma_1 \models d_1 \amalg T : \Gamma_2 \models d_2 &=& (\emptyset \models S \amalg \Gamma_1 \models d_1) \amalg \Gamma_2 \models d_2 \\
\end{align}
\]

These 4 cases are the same as before. The remaining cases prepare the ground for lazy weakening.

\[
\begin{align}
  \emptyset \models d_1 \amalg F : \Gamma \models d_2 &=& \emptyset \models d_1 \amalg \Gamma \models d_2 \\
  F : \Gamma \models d_1 \amalg \emptyset \models d_2 &=& \Gamma \models d_1 \amalg \emptyset \models d_2 \\
  F : \Gamma_1 \models d_1 \amalg T : \Gamma_2 \models d_2 &=& (\emptyset \models B \amalg \Gamma_1 \models d_1) \amalg \Gamma_2 \models d_2 \\
  T : \Gamma_1 \models d_1 \amalg F : \Gamma_2 \models d_2 &=& (\emptyset \models C \amalg \Gamma_1 \models d_1) \amalg \Gamma_2 \models d_2 \\
  F : \Gamma_1 \models d_1 \amalg F : \Gamma_2 \models d_2 &=& \Gamma_1 \models d_1 \amalg \Gamma_2 \models d_2
\end{align}
\]

They also complicate the connection with the specialized bracket abstraction:

\[
\begin{array}{c c l}
\ceil{z} &=& T : \emptyset \models I \\
\ceil{s e} &=& F : \Gamma \models d \quad \textrm{where } \Gamma \models d = \ceil{e} \\
\ceil{\lambda e} &=&
  \left\{ 
    \begin{array}{ c l }
      \Gamma \models d & \quad \textrm{if } T:\Gamma \models d = \ceil{e} \\
      \Gamma \models (\emptyset \models K \amalg \Gamma \models d) & \quad \textrm{if } F:\Gamma \models d = \ceil{e} \\
      \Gamma \models K d & \quad \textrm{if } \emptyset \models d = \ceil{e}
    \end{array}
  \right. \\
\ceil{e_1 e_2} &=& \Gamma_1 \vee \Gamma_2 \models (\Gamma_1 \models d_1 \amalg \Gamma_2 \models d_2) \\
& & \quad \textrm{where } \Gamma_1 \models d_1 = \ceil{e_1}, \Gamma_2 \models d_2 = \ceil{e_2}
\end{array}
\]

where $\vee$ means we OR together the corresponding booleans of each input
list, and if one list is longer than the other, we append the leftovers.

\begin{code}
convertBool (#) = \case
  N Z -> (True:[], Com "I")
  N (S e) -> (False:g, d) where (g, d) = rec $ N e
  L e -> case rec e of
    ([], d) -> ([], Com "K" :@ d)
    (False:g, d) -> (g, ([], Com "K") # (g, d))
    (True:g, d) -> (g, d)
  A e1 e2 -> (zipWithDefault False (||) g1 g2, t1 # t2) where
    t1@(g1, _) = rec e1
    t2@(g2, _) = rec e2
  Free s -> ([], Com s)
  where rec = convertBool (#)

optK = convertBool (#) where
  ([], d1) # ([], d2) = d1 :@ d2
  ([], d1) # (True:g2, d2) = ([], Com "B" :@ d1) # (g2, d2)
  ([], d1) # (False:g2, d2) = ([], d1) # (g2, d2)
  (True:g1, d1) # ([], d2) = ([], Com "R" :@ d2) # (g1, d1)
  (False:g1, d1) # ([], d2) = (g1, d1) # ([], d2)
  (True:g1, d1) # (True:g2, d2) = (g1, ([], Com "S") # (g1, d1)) # (g2, d2)
  (False:g1, d1) # (True:g2, d2) = (g1, ([], Com "B") # (g1, d1)) # (g2, d2)
  (True:g1, d1) # (False:g2, d2) = (g1, ([], Com "C") # (g1, d1)) # (g2, d2)
  (False:g1, d1) # (False:g2, d2) = (g1, d1) # (g2, d2)

zipWithDefault d f     []     ys = f d <$> ys
zipWithDefault d f     xs     [] = flip f d <$> xs
zipWithDefault d f (x:xt) (y:yt) = f x y : zipWithDefault d f xt yt
\end{code}

This corresponds with Kiselyov's OCaml implementation from Section 4, but we're
more lax with types, and we use a standard list of booleans rather than define
a dedicated type. Kiselyov's `C` can be thought of as the empty list, while `N`
and `W` add True and False to an existing list. It's like Peano arithmetic with
two kinds of successors.

To compute the OR of two lists, we clumsily traverse two lists with
`zipWithDefault` even though the recursive calls to `(#)` have already examined
the same booleans. We sacrificed elegance so that `(#)` corresponds to the
$\amalg$ operation. Kiselyov instead builds the output list while recursing.

As Kiselyov notes, lazy weakening is precisely link:cl.html[the famous
K-optimization of textbook bracket abstraction]. It fits snugly here, whereas
the bracket abstraction algorithms awkwardly require extra passes to detect the
absence of a variable in a subtree for each lambda.

== The Eta Optimization ==

The above postpones the conversion of $s$ to a $K$ combinator. Section 4.1 of
Kiselyov describes how to postpone the conversion of $z$ to an $I$ combinator.
The `V` that appears in his code is an IOU that is either eventually redeemed
for an $I$ combinator or optimized away.

We take a blunter approach. During the computation of $\amalg$, we simply match
on $I$ combinators that can be optimized. Loosely speaking, we examine the
result of a bracket abstraction to determine our course of action, rather than
pass around what we need to know on the left of a $\models$.

This may be messy, but on the other hand, we have no need to extend the list of
booleans to a more complex data type, and we simply omit the `(N e, V)` and
`(V, N e)` cases where no optimizations apply.

Here, we prefer the combinator $T$ given by $Tab = ba$ to the $CI$ in the
paper.

\begin{code}
optEta = convertBool (#) where
  ([], d1) # ([], d2) = d1 :@ d2
  ([], d1) # (True:[], Com "I") = d1
  ([], d1) # (True:g2, d2) = ([], Com "B" :@ d1) # (g2, d2)
  ([], d1) # (False:g2, d2) = ([], d1) # (g2, d2)
  (True:[], Com "I") # ([], d2) = Com "T" :@ d2
  (True:[], Com "I") # (False:g2, d2) = ([], Com "T") # (g2, d2)
  (True:g1, d1) # ([], d2) = ([], Com "R" :@ d2) # (g1, d1)
  (True:g1, d1) # (True:g2, d2) = (g1, ([], Com "S") # (g1, d1)) # (g2, d2)
  (True:g1, d1) # (False:g2, d2) = (g1, ([], Com "C") # (g1, d1)) # (g2, d2)
  (False:g1, d1) # ([], d2) = (g1, d1) # ([], d2)
  (False:g1, d1) # (True:[], Com "I") = d1
  (False:g1, d1) # (True:g2, d2) = (g1, ([], Com "B") # (g1, d1)) # (g2, d2)
  (False:g1, d1) # (False:g2, d2) = (g1, d1) # (g2, d2)
\end{code}

== Bulk combinators ==

If we allow families of certain 'bulk combinators', we get a linear-time
specialized bracket abstraction, which in turn yields a linear-time translation
algorithm from lambda terms to combinators.

Define:

\[
\begin{align}
B_n f g x_n ... x_1&=  f             &&(g x_n ... x_1)  \\
C_n f g x_n ... x_1&=  f x_n ... x_1 &&g \\
S_n f g x_n ... x_1&=  f x_n ... x_1 &&(g x_n ... x_1)  \\
\end{align}
\]

These perfectly suit shared-eta bracket abstraction:

\begin{code}
bulkPlain bulk = convert (#) where
  (a, x) # (b, y) = case (a, b) of
    (0, 0)             ->               x :@ y
    (0, n)             -> bulk "B" n :@ x :@ y
    (n, 0)             -> bulk "C" n :@ x :@ y
    (n, m) | n == m    -> bulk "S" n :@ x :@ y
           | n < m     ->                      bulk "B" (m - n) :@ (bulk "S" n :@ x) :@ y
           | otherwise -> bulk "C" (n - m) :@ (bulk "B" (n - m) :@  bulk "S" m :@ x) :@ y

bulk c 1 = Com c
bulk c n = Com (c ++ show n)
\end{code}

We write a version with K-optimization and eta-optimization, the latter
generalized to bulk $B_n$ combinators. Again we replace counts with lists of
booleans indicating whether a variable is used in a term. To exploit bulk
combinators, we look for repetitions of the same booleans.

This time, our `(##)` function returns the list of booleans along with the
combinator term, so the caller need not compute it.

\begin{code}
bulkOpt bulk = \case
  N Z -> (True:[], Com "I")
  N (S e) -> first (False:) $ rec $ N e
  L e -> case rec e of
    ([], d) -> ([], Com "K" :@ d)
    (False:g, d) -> ([], Com "K") ## (g, d)
    (True:g, d) -> (g, d)
  A e1 e2 -> rec e1 ## rec e2
  Free s -> ([], Com s)
  where
  rec = bulkOpt bulk
  ([], d1) ## ([], d2) = ([], d1 :@ d2)
  ([], d1) ## ([True], Com "I") = ([True], d1)
  ([], d1) ## (g2, Com "I") | and g2 = (g2, bulk "B" (length g2 - 1) :@ d1)
  ([], d1) ## (g2@(h:_), d2) = first (pre++) $ ([], fun1 d1) ## (post, d2)
    where
    fun1 = case h of
      True -> (bulk "B" (length pre) :@)
      False -> id
    (pre, post) = span (h ==) g2

  ([True], Com "I") ## ([], d2) = ([True], Com "T" :@ d2)
  (g1@(h:_), d1) ## ([], d2) = first (pre++) $ case h of
    True -> ([], Com "C" :@ bulk "C" (length pre) :@ d2) ## (post, d1)
    False -> (post, d1) ## ([], d2)
    where
    (pre, post) = span (h ==) g1

  ([True], Com "I") ## (False:g2, d2) = first (True:) $ ([], Com "T") ## (g2, d2)
  (False:g1, d1) ## ([True], Com "I") = (True:g1, d1)
  (g1, d1) ## (g2, Com "I") | and g2, let n = length g2, all not $ take n g1 =
    first (g2++) $ ([], bulk "B" $ n - 1) ## (drop n g1, d1)
  (g1, d1) ## (g2, d2) = pre $ fun1 (drop count g1, d1) ## (drop count g2, d2)
    where
    (h, count) = headGroup $ zip g1 g2
    fun1 = case h of
      (False, False) -> id
      (False, True) -> apply "B"
      (True, False) -> apply "C"
      (True, True) -> apply "S"
    pre = first (replicate count (uncurry (||) h) ++)
    apply s = (([], bulk s count) ##)

headGroup (h:t) = (h, 1 + length (takeWhile (== h) t))
\end{code}

Classic bracket abstraction algorithms blow up quadratically on lambdas terms
of the form:

\[ \lambda ... \lambda 0 ... n \]

In contrast, our implementation finds:

\[ C_n (...(C_2 T)) \]

== No bulk combinators ==

What if we disallow families of combinators?

If we allow sharing (memoization), then define the following cousins of $B, C,
S$ (Smullyan might call them "once removed"):

\[
\begin{align}
B'd f g x & = d f (g x) \\
C'd f g x & = d (f x) g \\
S'd f g x & = d (f x) (g x)
\end{align}
\]

The eta-optimized translation yields:

\[
\begin{align}
B' & = BB \\
C' & = B(BC)B \\
S' & = B(BS)B
\end{align}
\]

We have:

\[
\begin{align}
B_{n+1} & = B' B_n \\
C_{n+1} & = C' C_n \\
S_{n+1} & = S' S_n
\end{align}
\]

We can use these recurrences to generate each bulk combinator we need exactly
once, again resulting in a linear-time translation:

\begin{code}
breakBulkLinear "B" n = iterate (comB' :@) (Com "B") !! (n - 1)
breakBulkLinear "C" n = iterate (comC' :@) (Com "C") !! (n - 1)
breakBulkLinear "S" n = iterate (comS' :@) (Com "S") !! (n - 1)

comB' = Com "B":@ Com "B"
comC' = Com "B":@ (Com "B" :@ Com "C"):@ Com "B"
comS' = Com "B":@ (Com "B" :@ Com "S"):@ Com "B"
\end{code}

Our interactive demo supports this translation, but shows the combinator term
expanded in full, with no sharing.

== No sharing ==

What if we also diasllow sharing?

We generalize $C'$ and $S'$ from the paper:

\[
\begin{align}
C'_n c f g x_n ... x_1 &= c (f x_n ... x_1) g  \\
S'_n c f g x_n ... x_1 &= c (f x_n ... x_1) (g x_n ... x_1) \\
\end{align}
\]

One can check:

\[
\begin{align}
B_{m+n} &= B B_m B_n  \\
C'_{m+n} &= B C'_m C'_n  \\
S'_{m+n} &= B S'_m S'_n  \\
\end{align}
\]

This suggests breaking bulk combinators into standard combinators in a manner
akin to https://en.wikipedia.org/wiki/Exponentiation_by_squaring[exponentiation
by squaring]. To compute $S_{50}$ for example, define the following
combinators:

\[
\begin{align}
b_0 x &= B x x  \\
b_1 x &= B S'_1 (B x x)  \\
\end{align}
\]

In combinators: $b_0 = SBI, b_1 = B(B(B(BS)B))(SBI)$.

Then chain together $b_0$ or $b_1$ according to the bits of 50 written in
binary (110010), and apply to $I$:

\[
S_{50} = b_0(b_1(b_0(b_0(b_1(S'_1)))))I
\]

The $B_n$ case is simpler because there's no need for $B'$.

\begin{code}
bits n = r:if q == 0 then [] else bits q where (q, r) = divMod n 2

breakBulkLog c 1 = Com c
breakBulkLog "B" n = foldr (:@) (Com "B") $ map (bs!!) $ init $ bits n where
  bs = [sbi, Com "B" :@ (Com "B" :@ Com "B") :@ sbi]
breakBulkLog c n = (foldr (:@) (prime c) $ map (bs!!) $ init $ bits n) :@ Com "I" where
  bs = [sbi, Com "B" :@ (Com "B" :@ prime c) :@ sbi]
  prime c = Com "B" :@ (Com "B" :@ Com c) :@ Com "B"

sbi = Com "S" :@ Com "B" :@ Com "I"
\end{code}

Thus we can break down bulk combinators into standard combinators without
sharing if we pay a factor logarithmic in the input size.

The savings are more easily seen with the following demo, which breaks down a
single bulk combinator.

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
<p>Bulk combinator: <textarea id="inputbb" rows="1" cols="40">S50</textarea></p>
<button id="linearbb">Linear</button>
<button id="logbb">Logarithmic</button>
<p><textarea id="outputbb" rows="3" cols="80" readonly></textarea></p>
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

== Return of the Combinators? ==

Combinators were abandoned by compiler authors long ago: how do you optimize a
combinator term? It seems we must simulate it in order to undertand it enough
to tinker with it, but doesn't this simulation convert it back to a lambda
term? See Peyton Jones,
https://www.microsoft.com/en-us/research/wp-content/uploads/1987/01/slpj-book-1987-small.pdf['The
Implementation of Functional Programming Languages'], section 16.5.

Kiselyov suggests combinators may be worth considering again. Unlike existing
approaches, his algorithm understands lambda terms, so it might be suitable for
optimizations that are conventionally performed after generating combinations,
thus avoiding simulations. At any rate, on real world examples, the
eta-optimized variant seems to produce reasonable results.

link:crazyl.html[Our toy compiler for various minimalist combinatory logic and
lambda calculus languages] features Kiselyov's algorithm.

So does my link:../compiler/[award-winning Haskell compiler], which built the
wasm binaries running the widgets on this page.
