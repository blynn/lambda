= Combinatory Logic =

[pass]
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
<p>Lambda term: <textarea id="input" rows="1" cols="40">\f.(\x.x x)(\x.f(x x))</textarea></p>

Look ma,
<button id="debruijn">no names!</button>
<button id="rosenbloom">no variables!</button>
<button id="optimizeK">no variables, K-optimized!</button>
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

  go("debruijn", "input", "output");
  go("rosenbloom", "input", "output");
  go("optimizeK", "input", "output");
  go("rosenbloomnox", "inputnox", "outputnox");
  go("optimizeknox", "inputnox", "outputnox");
  go("stepsk", "iosk", "iosk");
  go("whnfsk", "iosk", "iosk");
  go("normsk", "iosk", "iosk");
}

WebAssembly.instantiateStreaming(fetch('cl.wasm'), { env:
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
foreign export ccall "debruijnMain" debruijnMain
debruijnMain = interact $ either id (show . deBrujin) . parseLC
foreign export ccall "rosenbloomMain" rosenbloomMain
rosenbloomMain = interact $ either id (show . nova rosenbloom) . parseLC
foreign export ccall "optimizeKMain" optimizeKMain
optimizeKMain = interact $ either id (show . nova optimizeK) . parseLC
foreign export ccall "rosenbloomnoxMain" rosenbloomnoxMain
rosenbloomnoxMain = interact $ either id (show . rosenbloom "x") . parseCL
foreign export ccall "optimizeknoxMain" optimizeknoxMain
optimizeknoxMain = interact $ either id (show . optimizeK "x") . parseCL
foreign export ccall "stepskMain" stepskMain
stepskMain = interact $ either id (show . step) . parseCL
foreign export ccall "whnfskMain" whnfskMain
whnfskMain = interact $ either id (show . whnf) . parseCL
foreign export ccall "normskMain" normskMain
normskMain = interact $ either id (show . norm) . parseCL

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

isAlphaNum c
  | 'A' <= c && c <= 'Z' = True
  | 'a' <= c && c <= 'z' = True
  | '0' <= c && c <= '9' = True
  | otherwise = False

lcTerm = lam <|> app where
  lam = flip (foldr Lam) <$> (lam0 *> some var <* lam1) <*> lcTerm
  lam0 = str "\\" <|> str "\955"
  lam1 = str "->" <|> str "."
  app = foldl1 App <$> some (Var <$> var <|> str "(" *> lcTerm <* str ")")
  var = some (sat isAlphaNum) <* whitespace

clTerm = app where
  app = foldl1 (:@) <$> some (com <|> var <|> str "(" *> clTerm <* str ")")
  com = Com <$> (str "S" <|> str "K")
  var = Tmp <$> some (sat isAlphaNum) <* whitespace

str = (<* whitespace) . string
whitespace = many $ char ' '

parseLC = fmap fst . getCharser (whitespace *> lcTerm <* eof)
parseCL = fmap fst . getCharser (whitespace *> clTerm <* eof)

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

\begin{code}
data Term = Var String | App Term Term | Lam String Term
\end{code}

Variables are the trickiest part of lambda calculus. And naming is the
trickiest part of variables: the most complex code in our lambda evaluator is
the part that renames variables to perform capture-avoiding substitutions.

Also, the lambda term $\lambda x \lambda y . y x$ is really the same as
$\lambda y \lambda z . z y$, but we must rename variables in order to see this.

Names seem artificial; tedious tags solely to aid human comprehension. Can we
get rid of them? There ought to be a way to study computation without naming
names.

(We will shortly mention many names, but they will be the names of researchers
and not variables!)

== De Bruijn ==

Large institutions like universities often anonymize by replacing names with
numbers. We do the same. We replace each occurrence of a variable with a
natural number indicating which lambda bound it, known as its
https://en.wikipedia.org/wiki/De_Bruijn_index['De Bruijn index'].

The innermost lambda is 0 (though some heretics prefer to start from 1), and
we increment as we go outwards.
http://www.cs.ox.ac.uk/richard.bird/online/BirdMeertens98Nested.pdf[Numbering
the other way is possible], but then common subexpressions appear different.

Instead of a sequence of bound variables like $\lambda x y z$, we now write a
string of repeated lambdas like $\lambda \lambda \lambda$. This means we no
longer we need a period to separate the lambda bindings from the body of an
expression.

For example,

\[
\lambda f.(\lambda x.x x)(\lambda x.f(x x))
\]

becomes:

\[
\lambda(\lambda 0 0)(\lambda 1(0 0))
\]

\begin{code}
data DB = N Int | L DB | A DB DB

instance Show DB where
  showsPrec p = \case
    N n -> (if p > 0 then (' ':) else id) . shows n
    L t -> ('\955':) . shows t
    A t u -> showParen (p > 0) $ showsPrec 0 t . showsPrec 1 u

index x xs = lookup x $ zip xs [0..]

deBrujin = go [] where
  go binds = \case
    Var x -> maybe (error $ "free: " ++ x) N $ index x binds
    Lam x t -> L $ go (x:binds) t
    App t u -> A (go binds t) (go binds u)
\end{code}

We port our lambda term evaluator to De Bruijn indices. Gone is the complex
renaming logic, and basic arithmetic suffices for telling apart free and bound
variables.

\begin{code}
whnfDB t = case t of
  A f x -> case whnfDB f of
    L body -> whnfDB $ beta 0 x body
    g -> A g x
  L _ -> t
beta lvl x = \case
  N n | n < lvl   -> N n
      | n == lvl  -> x
      | otherwise -> N $ n - 1
  A t u -> A (beta lvl x t) (beta lvl x u)
  L t -> L $ beta (lvl + 1) x t
\end{code}

== Schönfinkel, Curry ==

Buoyed by our success, we aim higher. Can we remove variables altogether,
whether they're names or numbers?

We start with a simple demonstration. Define:

\[
I = \lambda x . x = \lambda 0
\]

and:

\[
K = \lambda x y . x = \lambda \lambda 1
\]

Then we find for any $a, b$:

\[
KI = (\lambda \lambda x y.x) (\lambda x.x) a b = (\lambda x.x) b = b
\]

Thus $KI = \lambda x y . y = \lambda \lambda 0$. In other words, we have
combined $K$ and $I$ to produce a closed lambda term distinct to either one.
(Thanks to De Bruijn indices, we can see at a glance they differ.)
Moreover, no variables were needed: we merely applied $K$ to $I$.

Can we choose a small set of closed lambda terms such that they can be applied
to one another to produce any given closed lambda term?

Moses Schönfinkel asked and answered this question when he presented the ideas
of
http://www.cip.ifi.lmu.de/~langeh/test/1924%20-%20Schoenfinkel%20-%20Ueber%20die%20Bausteine%20der%20mathematischen%20Logik.pdf['On
the building blocks of mathematical logic'] to the Göttingen Mathematical
Society on 7 December 1920. It was written up for publication in March 1924 by
Heinrich Behmann.

A few years later, Haskell Curry rediscovered similar ideas, and only became
aware of Schönfinkel's work in late 1927. His solution was slightly different.

We say 'combinator' rather than "closed lambda term" in this context, which
perhaps reminds us that we're focused on combining specific closed lambda terms
to see what we can build, rather than playing games with variables.

== Church ==

All this was years before lambda calculus were discovered! How is this
possible? Recall any boolean circuit can be constructed just from NAND gates.
Schönfinkel sought to analogously reduce predicate logic into as few elements
as possible, and it turns out the "for all" and "there exists" quantifiers of
predicate logic behave like lambda abstractions.

Logic also motivated Curry, who wanted to better understand the rule of
substitution. Even Russell and Whitehead's 'Principia Mathematica', famed
for an excruciatingly detailed proof of $1 + 1 = 2$, lacked an explicit
definition of substitution.

In fact, around 1928 Alonzo Church devised lambda calculus so he could
explicitly state and analyze substitution, so it's not that Schönfinkel and
Curry traveled in time, but that everyone was studying the same hot topic.
Due to its origin story, the study of combinators is called 'combinatory
logic'.

Schönfinkel wrote definitions like $Ix = x$ instead of $I = \lambda x.x$ which
Church would later invent. We will too, to set the mood.
(Decades later, https://www.cs.cmu.edu/~crary/819-f09/Landin66.pdf[Landin
reverted to Schönfinkel's style to avoid frightening readers with lambdas]!)

Schönfinkel found that all combinators, that is, all closed lambda terms, can
be built exclusively from the $S$ and $K$ combinators, which are given by:

\[
\begin{aligned}
  & Sxyz = xz(yz) \\
  & Kxy = x
\end{aligned}
\]

except he wrote "C" instead of "K" (which is strange because the paper calls it
the 'Konstanzfunktion' so you'd think "K" would be preferred;
https://blog.plover.com/math/combinator-s.html[Curry likely changed the letters
to what we write today]). By the way, in Haskell $S$ is `(<*>)` (specialized
to Reader) and $K$ is `const`.

We define a data structure to hold expressions built from combinators. It's
little more than a binary tree, because we no longer have lambda abstractions.
A wrinkle is that we support storing a variable in a leaf. It turns out we
temporarily need to do this sometimes, hence the name `Tmp`.

\begin{code}
data CL = Com String | Tmp String | CL :@ CL

instance Show CL where
  showsPrec p = \case
    Com s -> (if p > 0 then (' ':) else id) . (s++)
    Tmp s -> (if p > 0 then (' ':) else id) . (s++)
    t :@ u -> showParen (p > 0) $ showsPrec 0 t . showsPrec 1 u
\end{code}

A cleaner design may be to import the `Data.Magma` package and use `BinaryTree
String` to hold combinator terms and `BinaryTree (Either String String)` to
hold combinator terms that may contain variables. But that's more code than
I want for simple demo.

Schönfinkel went further and combined $S$ and $K$ into a single
mega-combinator; the https://en.wikipedia.org/wiki/Iota_and_Jot[Iota and Jot
languages] do something similar. We seem to gain no advantage from these
contrived curiosities, other than being able to brag that one combinator
suffices for any computation.

== Rosenbloom ==

So how did Schönfinkel convert any closed lambda term to a tree of $S$ and $K$
combinators? We won't say, firstly because Schönfinkel didn't actually state an
algorithm and only walked through an example, and secondly because we prefer to
examine a more streamlined version published by Paul Rosenbloom in 1950 (page
117 of https://stacks.stanford.edu/file/druid:xm010sf8035/xm010sf8035.pdf['The
Elements of Mathematical Logic']).

The key is to solve a subproblem known as 'bracket abstraction'.
Suppose we have an expression $e$ built from $S$ and $K$ combinators and
variables, for example:

\[
e = S K (S x) (y K z)
\]

Pick a variable, say $x$. Then find an expression $f$ such that

\[ f x = e \]

where $f$ does not contain any occurrences of $x$ (so $f$ is built from $S, K$
and variables other than $x$). Smullyan calls $f$ an '$x$-eliminate of $e$' in
his puzzle book 'To Mock a Mockingbird'. Some authors write:

\[ f = [x] e \]

I believe this problem is so named because some authors use brackets to denote
substitution. For example, the notation $[P/x] x = P$ means "substituting $P$
for $x$ in the term $x$ results in $P$".

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
<p>[x] <textarea id="inputnox" rows="1" cols="40">S K (S x)(y K z)</textarea></p>
<button id="rosenbloomnox">Rosenbloom</button>
<button id="optimizeknox">K-optimized</button>
<p><textarea id="outputnox" rows="3" cols="80" readonly></textarea></p>
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Like just about all problems involving trees, recursion is the solution.

If $e = x$, then a solution is $f = SKK$. We can check $SKKx = x$.

If $e = y$, then a solution is $f = Ky$, as $Kyx = y$. This works for any
variable apart from $x$, or $S$ or $K$.

Otherwise $e = t u$ where $t, u$ are expressions. Then we recursively solve
the problem for $t, u$ to get $t', u'$, and a solution is $f = S t' u'$.
We have $S t' u' x = t' x (u' x) = t u$.

\begin{code}
rosenbloom x = \case
  Tmp y | x == y -> Com "S" :@ Com "K" :@ Com "K"
  t :@ u -> Com "S" :@ rosenbloom x t :@ rosenbloom x u
  t -> Com "K" :@ t
\end{code}

https://www.cantab.net/users/antoni.diller/brackets/intro.html[Antoni Diller's
page on classic bracket abstraction] features online interactive demos of many
variants of this algorithm.

Converting a given closed lambda term to $S$ and $K$ combinators is simply
a matter of performing bracket abstraction for each lambda. We parameterize the
bracket abstraction function so we can play with different versions.

\begin{code}
nova elim = \case
  App t u -> rec t :@ rec u
  Lam x t -> elim x $ rec t
  Var x -> Tmp x
  where rec = nova elim
\end{code}

Naturally, we can run $SK$ programs in link:.[our lambda evaluator]
by adding the definitions:

------------------------------------------------------------------------
S = \x y z.x z(y z)
K = \x y.x
------------------------------------------------------------------------

However, we wish to flaunt the simplicity of $SK$ interpreters. Here are
functions for head reduction, weak head normalization, and normaliation:

\begin{code}
reduce (Com "S" :@ x :@ y :@ z) = Just $ x :@ z :@ (y :@ z)
reduce (Com "K" :@ x :@ y) = Just x
reduce _ = Nothing

step (f :@ z) = maybe (step f :@ z) id $ reduce (f :@ z)
step t = t

whnf (f :@ z) = maybe t whnf $ reduce t where t = whnf f :@ z
whnf t = t

norm t = case whnf t of
  x :@ y -> norm x :@ norm y
  t -> t
\end{code}

Clearly, evaluating $S$ and $K$ combinators is far simpler than dealing with
beta-reduction in lambda calculus even if we employ De Bruijn indices.

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
<p>SK program:
<textarea id="iosk" rows="3" cols="80">S K K (K (S K K K))</textarea></p>
<p>
<button id="stepsk">Step</button>
<button id="whnfsk">Weak Head Normalize</button>
<button id="normsk">Normalize</button>
</p>
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

== Turner ==

Together, the $S$ and $K$ combinators possess as much power as lambda calculus,
that is, they are Turing complete! Thus a good strategy for a compiler might be
to translate a given lambda calculus program to combinators because it's so
easy to generate code that repeatedly reduces $S$ and $K$ combinators.

However, there's a snag. This idea only works if the conversion is efficient.
Rosenbloom's bracket abstraction algorithm is exponential in the worst case.
Consider, say:

\[
\lambda x_1 ... x_n . S K
\]

We start with one application, of $S$ to $K$. For each lambda, the `rosenbloom`
function recursively calls itself on either side of each application, which at
least doubles the number of applications, so the result will have at least
$2^n$ applications.

A few tweaks greatly improve the efficiency, and indeed, Schönfinkel's paper
suggests a faster algorithm, and Curry explicitly stated more efficient
algorithms.

A particular effective improvement known as the 'K-optimization' is to detect
an unused variable as far up the tree as possible and apply $K$ here, rather
than painstakingly send the variable down to each leaf of the tree, only to
have it eventually ignored via $K$ at the last minute.

\begin{code}
occurs x = \case
  Tmp s | s == x -> True
  t :@ u -> occurs x t || occurs x u
  _ -> False

optimizeK x t
  | not $ occurs x t = Com "K" :@ t
  | otherwise = case t of
    Tmp y -> Com "S" :@ Com "K" :@ Com "K"
    t :@ u -> Com "S" :@ optimizeK x t :@ optimizeK x u
\end{code}

David Turner found enough optimizations to enable a practical
https://www.cs.kent.ac.uk/people/staff/dat/miranda/downloads/[compiler for his
Miranda language], that is, we implement a small set of combinators, then
rewrite the input program in terms of those combinators.

Early Haskell compilers took the same approach, and later Haskell compilers
extended this approach by generating custom combinators that are tailor-made
for each input program. These are known as 'supercombinators' in the
literature. However, modern GHC shuns combinators in favour of
https://www.microsoft.com/en-us/research/wp-content/uploads/1992/04/spineless-tagless-gmachine.pdf[the
Spineless Tagless G-Machine].

See the effectiveness of this approach in link:sk.html[our
lambda-calculus-to-wasm compiler].

== Iverson ==

Combinators are also useful on the other side of the compiler, that is, in
source programs thesmelves. Variables can be distracting at times, and code may
be clearer if it describes what it is doing without mentioning variables. This
is known as https://en.wikipedia.org/wiki/Tacit_programming[point-free style or
tacit programming].

Ken Iverson's APL language pioneered this technique, and developed it to such
an extent that it avoiding variables became idiomatic. John Backus was
inspired, as can be observed in
http://www.dgp.toronto.edu/~wael/teaching/324-03f/papers/backus.pdf[his
renowned Turing award lecture], though oddly, he accuses the closely related
lambda calculus of fomenting chaos; perhaps he was overly enamoured by APL's
informal ban on variables!

== Mac Lane, Eilenberg ==

In mathematics, the study of functions without mentioning variables occurs in
https://en.wikipedia.org/wiki/Category_theory[category theory], founded by
Saunders Mac Lane and Samuel Eilenberg. The absence of variables not only
simplifies the picture, but also enables generalization: we can make sweeping
statements that apply to sets and functions as well as, say, posets.

Conal Elliott, http://conal.net/papers/compiling-to-categories/['Compiling to
categories'], explores the connection between point-free code and category
theory.

My main sources for the historical details were:

  * Cardone and Hindley, http://www.users.waitrose.com/~hindley/SomePapers_PDFs/2006CarHin,HistlamRp.pdf['History of Lambda-calculus and Combinatory Logic']
  * van Heijenoort, 'From Frege to Gödel'
