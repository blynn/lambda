= Self-aware Computing =

How can we compose two Turing machines? It seems reasonable answers boil down
to running one after the other.

How about composing two lambda calculus terms? This time, the answers boil down
to applying one to the other.

These observations sound similar, but there is a difference. A Turing machine
can only learn about another Turing machine via symbols on the tape. There is
no way for the second machine to learn any internal details of the first
machine.

In contrast, to apply a lambda term to another is to supply an entire program
as the input to another. This raises a question we cannot ask about Turing
machines: can a cleverly designed lambda term tell us anything interesting
about another lambda term?

Yes! http://www.di.unito.it/~dezani/papers/dgp.pdf[Böhm's Theorem] involves an
algorithm that, given any two distinct closed normal lambda terms $A$ and
$B$, builds a term $X$ such that $XA$ is true and $XB$ is false.

By true and false we mean lambda terms representing booleans. Which
representation? It doesn't matter! We may choose any two lambda terms to
represent true and false.

Our demo uses Church booleans:

[pass]
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
<script>
function hideshow(s) {
  const x = document.getElementById(s);
  x.style.display = x.style.display === "none" ? "block" : "none";
}
</script>
<p>
<button onclick="hideshow('eg')">Examples</button>
</p>
<style>#eg {
position:absolute;
background-color:white;
border:1px solid grey;
padding:2px;
}#eg div:hover{
background-color:grey;
}
</style>
<div id="eg"></div>
<p>A: <textarea id="inputa" rows="1" cols="40">
</textarea></p>
<p>B: <textarea id="inputb" rows="1" cols="40">
</textarea></p>
<button onclick="go()">Böhm</button>
<p id="output" rows="12" cols="80">
</p>
<script>
"use strict";
var bohm;
var idx;
var cursor;
var inputs;
var out;
const eg = document.getElementById("eg");
const inputa = document.getElementById("inputa");
const inputb = document.getElementById("inputb");
WebAssembly.instantiateStreaming(fetch('bohm.wasm'), { env:
  { getchar: () => {
      if (inputs[idx].length == cursor) throw "eof";
      cursor++;
      return inputs[idx].charCodeAt(cursor - 1);
    }
  , putchar: c => out += String.fromCharCode(c)
  , eof: () => cursor == inputs[idx].length
  , nextinput: () => { idx++; cursor = 0; }
  }}).then(obj => {
    bohm = obj.instance;
    diff(exs[0], exs[1]);
  });

function readInputs() {
  idx = 0;
  cursor = 0;
  inputs = [inputa.value, inputb.value];
}

function go() {
  readInputs();
  out = "";
  bohm.exports.go();
  document.getElementById('output').innerHTML = out;
}

const exs =
  [ "\\x y z -> x z (y z)", "\\x y -> x"
  , "\\x y z -> x z x", "\\x y -> y"
  , "\\x -> x", "\\x y z w -> x y"
  , "\\x y -> x (y (x (y x)))", "\\x y -> x (y (x (y y)))"
  , "\\a b c d -> a (b (a a (b c b) a))", "\\a b c d -> a (b (a a (b d b) a))"
  , "\\x -> x (x x x) x", "\\x -> x (x x x x x x x) x"
  , "\\x -> x x (x (x x x) x)", "\\x -> x x (x (x x x x x x x) x)"
  , "\\u v -> u u (v v (u u (u u)))", "\\u v -> u u (v v (u u (v v)))"
  , "\\u v -> u u (v v (u u (u u)))", "\\u v -> u u (v v (u u v))"
  , "\\a b -> a (\\c x -> b (\\d -> a b d) x)", "\\a b -> a (\\c -> b (\\d y -> a b c y))"
  ]

function diff(a, b) {
  inputa.value = a;
  inputb.value = b;
  eg.style.display = "none";
  go();
}

for(let i = 0; i < exs.length; i += 2) {
  const newDiv = document.createElement("div");
  const a = exs[i];
  const b = exs[i + 1];
  newDiv.innerText = a + " versus " + b
  newDiv.addEventListener("click", function(){diff(a, b);});
  eg.appendChild(newDiv);
}
</script>
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Perhaps "self-aware" is overselling it, but it's striking that in the primitive
world of lambda calculus, one program can examine another. A sliver of the
meta-theory has found its way into the theory.

== Separate but equal ==

We should clarify when lambda terms count as distinct.

We use De Bruijn indices to avoid games with variable names.
(Originally, variable names were part of the theory and renaming was dealt
formally via 'alpha-conversion'.)

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
<p><a onclick='hideshow("boilerplate");'>&#9654; Toggle boilerplate</a></p>
<div id='boilerplate' style='display:none'>
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
\begin{code}
{- GHC edition:
{-# LANGUAGE BlockArguments, LambdaCase #-}
import Control.Applicative
import Control.Arrow
import Data.Char
import Data.List
import Data.Function
import Data.Foldable
-}

module Main where
import Base

foreign import ccall "nextinput" nextInput :: IO ()
foreign import ccall "putchar" putChar :: Char -> IO ()
foreign import ccall "getchar" getChar :: IO Char
foreign import ccall "eof" isEOFInt :: IO Int

isEOF = (0 /=) <$> isEOFInt
putStr = mapM_ putChar
putStrLn = (>> putChar '\n') . putStr
print = putStrLn . show
getContents = isEOF >>= \b -> if b then pure [] else getChar >>= \c -> (c:) <$> getContents

isLower c = 'a' <= c && c <= 'z'
isUpper c = 'A' <= c && c <= 'Z'
isAlphaNum c = isLower c || isUpper c || '0' <= c && c <= '9'
max a b = if a <= b then b else a
maximum = foldr1 max
sortOn _ [] = []
sortOn f (x:xt) = sortOn f (filter ((<= fx) . f) xt)
  ++ [x] ++ sortOn f (filter ((> fx) . f) xt) where fx = f x
\end{code}
++++++++++
</div>
++++++++++

\begin{code}
data LC = Va Int | La LC | Ap LC LC
instance Show LC where
  showsPrec prec = \case
    Va n -> shows n
    Ap x y -> showParen (prec > 0) $ showsPrec 0 x . (' ':) . showsPrec 1 y
    La t -> showParen (prec > 0) $ ("&#955;"++) . shows t

data Expr = Expr :@ Expr | V String | L String Expr

debruijn bnd = \case
  V v -> maybe (Left $ "free: " ++ v) Right $ lookup v $ zip bnd $ Va <$> [0..]
  x :@ y -> Ap <$> debruijn bnd x <*> debruijn bnd y
  L s t -> La <$> debruijn (s:bnd) t
\end{code}

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
<p><a onclick='hideshow("parseeval");'>&#9654; Toggle parser and evaluator</a></p>
<div id='parseeval' style='display:none'>
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

\begin{code}
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

next = Charser \case
  [] -> Left "got EOF"
  h:t -> Right (h, t)

sat f = Charser \case
  h:t | f h -> Right (h, t)
  _ -> Left "unsat"
char c = sat (c ==)

whitespace = many $ sat isSpace

eof = Charser \case
  [] -> Right ((), "")
  _ -> Left "want EOF"

spch c = char c <* whitespace

var = ((:) <$> sat isLower <*> many (sat isAlphaNum)) <* whitespace

expr = (&) <$> aexp <*> ((\o r l -> o :@ l :@ r) <$> op <*> expr <|> pure id) where
  op = V <$> some (sat (`elem` ":=.")) <* whitespace
  aexp = foldl1 (:@) <$> some atom
  atom = spch '(' *> (op <|> expr) <* spch ')'
    <|> V <$> var
    <|> spch '\\'
      *> (flip (foldr L) <$> many var <*> (arrow *> expr))
  arrow = char '-' *> spch '>'

parse name s = case getCharser (whitespace *> expr <* eof) s of
  Left err -> Left $ name ++ ": " ++ err
  Right (x, _) -> Right x

norm = \case
  Ap f x -> case norm f of
    La t -> norm $ beta 0 x t
    notLa -> Ap notLa $ norm x
  La t -> La $ norm t
  t -> t
beta k x = \case
  Ap a b -> Ap (beta k x a) $ beta k x b
  La t -> La $ beta (k + 1) x t
  Va n | k == n -> freeVarPlus k x
       | k < n -> Va $ n - 1
       | otherwise -> Va n
freeVarPlus k = go 0 where
  go m = \case
    Ap a b -> Ap (go m a) (go m b)
    La t -> La $ go (m + 1) t
    Va n | n >= m -> Va $ n + k
         | otherwise -> Va n
\end{code}

++++++++++
</div>
++++++++++

Consider the terms $(λ0)λ0$ and $λ0$, or `id id` and `id` in Haskell. Suppose
we only have "black-box access" to them, that is, the only non-trivial
operation is applying one of them to some lambda term. Afterwards, we only
have black-box access to the result.

Then the terms are indistinguishable. In real life, we might be able tell that
we're evaluating `id id` instead of `id` because it takes more time or memory,
but then we're stepping outside the system. Thus for Böhm's Theorem, we
consider these terms to be the same. More generally, any 'beta-conversion' is
invisible, that is, $(λX)Y$ and $X[Y/0]$ are 'beta-equivalent', where the
latter is crude notation for substituting $Y$ for every free 0 in $X$ and
decrementing all other free variables.

Now consider the terms $λ0$ and $λλ10$. In Haskell, these are `id` and `(\$)`,
and `id f x` and `(\$) f x` both evaluate to `f x` for any suitably typed `f`
and `x`. Thus in untyped lambda calculus the two terms act the same. (In
Haskell, we can detect a difference thanks to types: for example `(\$) ()` is
illegal while `id ()` is fine. But in untyped lambda calculus, anything goes.)

Accordingly, for Böhm's Theorem we must view $λ0$ and $λλ10$ as equal, and more
generally, there's no way to tell apart $λx.Fx$ and $F$ from their external
properties. Replacing one with the other is called a 'eta-conversion', so
treating such terms as equals is known as 'eta-equivalence'. Replacing the the
smaller one with the bigger one is more specifically called 'eta-expansion' or
'eta-abstraction', while the reverse is 'eta-reduction'.

While we're throwing around buzzwords, if two terms are indistiguishable from
their external properties, then we say they are 'extensionally equal'.

Do we need to beware of other strange equivalences? No! It turns out beta- and
eta-equivalence is all we need for the theorem to work.

A consequence is that in lambda calculus, there is nothing beyond beta- and
eta-equivalence. For suppose the normal terms $A$ and $B$ are distinct up to
beta- and eta-conversion, and we add a new law implying $A = B$. Then $XA = XB$
for any $X$ and via Böhm's theorem, we can construct $X$ to equate any two
terms under the new law, forcing the equivalence of all normal terms.

== A/B Testing ==

A normal term must have the form:

\[
\lambda  ...  \lambda v T_1 ... T_n
\]

where we have zero or more lambda abstractions and a head variable $v$ that is
applied to zero or more normal terms. If $n > 0$ and $v$ is instead a lambda
abstraction, we could beta-reduce further: a contradiction for normal terms.

This observation inspires the definition of a
https://en.wikipedia.org/wiki/B%C3%B6hm_tree['Böhm tree']. In our version, a
Böhm tree node contains the number of preceding lambdas, the head variable $v$
which starts its body, and $n$ child Böhm trees representing the normal terms
to which $v$ is applied, and form the remainder of its body.

\begin{code}
data Bohm = Bohm Int Int [Bohm] deriving Show

bohmTree :: LC -> Bohm
bohmTree = smooshLa 0 where
  smooshLa k = \case
    La t -> smooshLa (k + 1) t
    t    -> smooshAp [] t where
      smooshAp kids = \case
        Va n   -> Bohm k n $ bohmTree <$> kids
        Ap x y -> smooshAp (y:kids) x
        _      -> error "want normal form"
\end{code}

Let $A$ and $B$ be normal lambda terms. We recursively descend the Böhm trees
representing $A$ and $B$ starting from their roots, looking for a difference.

We have no concerns about beta-equivalence because the terms are normal.
However, we must be mindful of eta-equivalence. When comparing Bohm trees, if
one has fewer lambdas than the other, then we compensate by performing
eta-conversions to make up the difference: we add lambdas, renumber variables
in the existing children, and add new children corresponding to the new
lambdas.

For example, adding 3 lambdas to:

\[
λλ v T_1 ... T_n
\]

results in the eta-equivalent:

\[
λλλλλ v T'_1 ... T'_n 2 1 0
\]

where $T'_k$ is $T_k$ with every free variable increased by 3.
Let's write $λ^n$ to mean $n$ lambda abstractions in a row; above we could have
written $λ^5$.

\begin{code}
eta n (Bohm al ah akids) = Bohm (al + n) (ah + n) $
  (boostVa 0 n <$> akids) ++ reverse (($ []) . Bohm 0 <$> [0..n-1])
boostVa minFree i t@(Bohm l h kids) =
  Bohm l ((if minFree' <= h then (i+) else id) h) $ boostVa minFree' i <$> kids
  where minFree' = minFree + l
\end{code}

Henceforth we may assume the Bohm trees we're comparing have the same lambda
count $l$.

One possibility is that the head variables of our trees differ, say:

\[
A = λ^l v T_1 ... T_m
\]

versus:

\[
B = λ^l w S_1 ... S_n
\]

where $v \ne w$. Then for any terms $U, V$, substituting

  * $v \mapsto λ^m U$

  * $w \mapsto λ^n V$

reduces the body of $A$ to $U$ and the body of $B$ to $V$.

Otherwise the trees have the same head variable $v$. Suppose they have a different number of children:

\[
A = λ^l v T_1 ... T_m
\]

versus:

\[
B = λ^l v S_1 ... S_n
\]

with $m < n$. Consider the terms:

  * $v T_1 ... T_m a_1 ... a_{n-m}$

  * $v S_1 ... S_n a_1 ... a_{n-m}$

Then substituting:

  * $v \mapsto λ^{n+1} 0$

  * $a_1 \mapsto λ^{n-m} V$

  * $a_{n-m} \mapsto U$

reduces the body of $A$ to $U$ and the body of $B$ to $V$.

Our code calls these two kinds of differences `Func` deltas and `Arity` deltas
respectively. We use lists of length 2 to record the parts that differ. In both
cases, we store the variable, number of children, and a choice of $U$ or $V$.

\begin{code}
data Delta = Func [(Int, (LC, Int))] | Arity Int [(Int, LC)]
\end{code}

The remaining case is that $A$ and $B$ have the same head variable $v$ and same
number of children $n$.

If their children are identical, then $A = B$, otherwise we recurse to find $k$
such that the bodies of $A$ and $B$ are:

\[
v T_1 ... T_n
\]

versus:

\[
v S_1 ... S_n
\]

with $T_k \ne S_k$. Substituting the $k$-out-of-$n$ projection function for
$v$:

\[
v \mapsto λ^n (n - k)
\]

reduces the first to $T_k$ and the second to $S_k$.

The `diff` function recursively traverses 2 given Bohm trees to find a
difference, eta-converting along the way to equalize the number of lambdas.
It records the `Path` taken to reach a `Delta`.

Our code calls a projection a `Pick`, because it picks out the $k$th child.
We shall need the `Tuple` alternative later.

We also record the number of children of each node along the path as well
as the level of nesting within lambda abstractions.

\begin{code}
data Path = Pick Int | Tuple deriving Show

diff nest0 a@(tru, Bohm al ah akids) b@(fal, Bohm bl bh bkids)
  | al > bl = diff nest0 a (second (eta $ al - bl) b)
  | al < bl = diff nest0 (second (eta $ bl - al) a) b
  | ah /= bh = base $ Func [(nest - ah - 1, (tru, an)), (nest - bh - 1, (fal,bn))]
  | an > bn = diff nest0 b a
  | an < bn = base $ Arity (nest - ah - 1) [(an, tru), (bn, fal)]
  | otherwise = asum $ zipWith (induct . aidx . Pick) [1..] $
    zipWith go akids bkids
  where
  base delta = Just ((nest, delta), [])
  induct x = fmap $ second (x:)
  nest = nest0 + al
  aidx t = (nest - ah - 1, ((nest, an), t))
  bidx t = (nest - bh - 1, ((nest, bn), t))
  an = length akids
  bn = length bkids
  go ak bk = diff nest (tru, ak) (fal, bk)
\end{code}

We construct $X = λ 0 X_1 ... X_n$ such that $XA = U$ and $XB = V$ by building
the $X_i$ so that the desired substitutions arise. For some values of $i$, we
may find any $X_i$ will do, in which case we pick `λ0`. [It turns out our code
relies on our choice being normalizing; if we cared, we could instead introduce
a special normal node, and only after `norm` has been called do we replace such
nodes with any terms we like, normalizing or not.]

It seems to be little more than fiddly bookkeeping:

\begin{code}
laPow n x = iterate La x !! n

lcPath ((_, n), sub) = case sub of
  Pick k -> laPow n $ Va $ n - k
  Tuple -> laPow (n + 1) $ foldl Ap (Va 0) $ Va <$> reverse [1..n]

easy (tru, a) (fal, b) = case diff 0 (tru, bohmTree a) (fal, bohmTree b) of
  Nothing -> Left "equivalent lambda terms"
  Just ((vcount, delta), path) -> Right $ (maybe whatever id . (`lookup` subs) <$> [0..vcount-1]) ++ arityArgs
    where
    whatever = La $ Va 0
    subs = addDelta $ second lcPath <$> path
    addDelta = case delta of
      Arity v [_, (b, _)] -> ((v, laPow (b + 1) $ Va 0):)
      Func [(av, (t, an)), (bv, (f, bn))] -> ([(av, laPow an t), (bv, laPow bn f)]++)
    arityArgs = case delta of
      Arity v [(a, t), (b, f)] -> let d = b - a in
        [laPow d f] ++ replicate (d - 1) whatever ++ [t]
      _ -> []
\end{code}

And indeed, it works. But sadly, not all the time.

== Lambda Overload ==

Trouble arises when different substitutions target the same variable.
For example, consider the two terms:

  * $λxy.x (x y y (y y)) y$

  * $λxy.x (x y y (y y y y)) y$

We have no problem substituting a term for $y$ to exploit the `Arity`
difference, but how do we simultaneously apply $x \mapsto λ^2 1$ to pick the
first child of the root node and $x \mapsto λ^3 0$ to pick the third child of
the first child?

We recall the aphorism: "We can solve any problem by introducing an extra level
of indirection."
We eta-convert so all nodes with head variable $x$ have the same number
of children, and also more children than they had before.
Above, an $x$ node has at most 3 children, so we can eta-convert every $x$ node
to have exactly 4 children.

  * $λxyab.x (λc.x y y (y y) c) y a b$

  * $λxyab.x (λc.x y y (y y y y) c) y a b$

Then the substitution $x \mapsto λ^4 0 3 2 1$ results in the
following bodies of the two nodes:

  * $b (λc.c y y (y y)) y a$

  * $b (λc.c y y (y y y y)) y a$

The new head variables are unique by construction, hence we can apply
appropriate `Pick` substitutions to home in on the delta. In our example, we
want:

  * $b \mapsto λ^3 2$

  * $c \mapsto λ^3 0$

More generally, suppose we have conflicting substitutions for $x$. Consider
all nodes along the path to the delta with head variable $x$. Let $t$ be
maximum number of children of any of the nodes, or greater.

Then we eta-convert each such node so it has exactly $t + 1$ children and apply:

  * $x \mapsto λ^{t+1} 0 t (t - 1) ... 1$

This effectively replaces each head variable $x$ with a unique fresh variable.
We repeat the procedure for every overloaded variable.

Our code calls this a `Tuple` because it's the Scott encoding of a $t$-tuple.

== The Hard Hard Case ==

Above, our example showed two clashing `Pick` substitutions, and at first
glance, it seems the same trick should also work for a `Pick` interfering with
`Func` or `Arity`. For example:

  * $λx.x (x x) x$

  * $λx.x (x x x x x x) x$

We want $x \mapsto λ^2 1$ to `Pick` the first child of the root Bohm tree node,
but also $x \mapsto λ^6 0$ to act upon the `Arity` mismatch.

Following the steps above, we eta-convert with $t = 6$, substitute a `Tuple`
for $x$, substitute a `Pick` for one of the fresh variables, and a suitable
`Arity` substitution for another, and we're done.

It would seem a `Pick`-`Func` collision could be handled similarly. This is
almost right: the trick indeed works most of the time. However, there is a
wrinkle when there are two `Pick`-`Func` collisions. Consider:

  * $λxyz.x (y (x (y x)))$

  * $λxyz.x (y (x (y y)))$

Both $x$ and $y$ have conflicting substitutions between a `Pick` and a `Func`.
Each node has at most one child. If we naively set $t_x = t_y = 1$, then we
wind up with the substitutions:

  * $x \mapsto λ^2 0 1$

  * $y \mapsto λ^2 0 1$

But replacing $x$ and $y$ with the same thing means we can no longer
distinguish between the two terms; any further efforts are doomed to failure.

The solution is to insist $t_x \ne t_y$, say by choosing $t_x = 1, t_y = 2$.
Thus after eta-conversion:

  * $λxyza.x (λbc.y (λd.x (λef.y (λg h.x g h) e f) d) b c) a$

  * $λxyza.x (λbc.y (λd.x (λef.y (λg h i.y g h i) e f) d) b c) a$

That is, when $x$ is the head of a node, there are 2 children, and when $y$ is
the head, there are 3 children. After substitution, these terms are
distinguishable and each head variable is distinct, so we have successfully
reduced the problem to the easy case.

This subtlety adds to the complexity of the already intimidating `dedup`
function, whose duties include finding overloaded head variables,
eta-converting each node in the paths to the deltas, and inserting the
corresponding `Tuple` substitutions.

The first time we find a `Func` head variable $x$ is overloaded, we choose the
smallest possible $t$ and pass it around via the `avoid` parameter. If we later
learn the other `Func` head variable $y$ is also overloaded, we ensure the
second choice for $t$ is distinct by incrementing on encountering equality.

Our straighforward approach guarantees we handle the hard hard case, though
sometimes it may actually be fine to reuse the same $t$. Similarly, we
unconditionally apply the `Tuple` trick whenever the same variable appears more
than once in the path, even though this is unnecessary if all the substitutions
happen to be identical.

\begin{code}
dedup _ delta [] = []
dedup avoid delta (h@(v, ((nest, n), _)):rest)
  | length matches == 1 = h : dedup avoid delta rest
  | otherwise           = (v, ((nest + t - n + 1, t), Tuple)) :
    dedup (avoid <|> const t <$> isFunc) delta (etaFill v t $ h:rest)
  where
  matchDelta = case delta of
    Arity w as | v == w -> (maximum (fst <$> as):)
    Func fs -> maybe id ((:) . snd) $ lookup v fs
    _ -> id
  matches = matchDelta $ snd . fst . snd <$> (h : filter ((v ==) . fst) rest)
  tNaive = maximum matches
  isFunc = case delta of
    Func fs -> lookup v fs
    _ -> Nothing
  t = case const <$> avoid <*> isFunc of
    Just taken | tNaive == taken -> tNaive + 1
    _ -> tNaive

renumber f (w, ((nest, n), step)) = (f w, ((f nest, n), step))

etaFill v t path = go path where
  go [] = []
  go (h@(w, ((nest, n), step)):rest)
    | v == w = (nest + pad, ((nest + pad + 1, t), step)) :
      go (renumber boost <$> rest)
    | otherwise = h : go rest
    where
    pad = t - n
    boost x
      | x >= nest = x + pad + 1
      | otherwise = x
\end{code}

The first time I tried implementing the algorithm, I attempted to adjust the
`Func` and `Arity` deltas as I eta-converted the nodes with overloaded head
variables. However, I kept tripping up over special cases, and lost confidence
in this approach. [For example, if $x$ and $y$ are head variables of a `Func`
delta, and $x$ is overloaded but $y$ is not, then it turns out we must modify
the substitution for $y$ after eta-converting to deal with $x$.]

Our current strategy is simpler, at the cost of computing deltas twice and
normalizing temporary terms in intermediate steps.

  1. Find a delta between two given lambda terms $A, B$.

  2. Build a lambda term $Y$ so that $YA, YB$ are $A, B$ with `Tuple` and
  `Pick` substitutions applied so that all head variables are unique. All other
  variables are kept abstracted. The `mkArgs` helper in `hard` sees to this.

  3. We have now reduced the problem to the easy case, which we solve as
  before to obtain a list of terms to be applied to $YA$ or $YB$.

  4. Build our final answer $X$ from $Y$ and this list of terms.

\begin{code}
hard (tru, a) (fal, b) = case diff 0 (tru, bohmTree a) (fal, bohmTree b) of
  Nothing -> Left "equivalent lambda terms"
  Just ((_, delta), path) -> Right $ mkArgs 0 0 $ sortOn fst $ dedup Nothing delta path
    where
    mkArgs freeCnt k [] = (freeCnt, [])
    mkArgs freeCnt k ((v, rhs):rest) = second (((Va <$> [freeCnt..freeCnt+v-k-1]) ++) . (lcPath rhs:)) $ mkArgs (freeCnt + v - k) (v + 1) rest

distinguish a b = do
  (freeCnt, uniqArgs) <- hard (churchTrue, a) (churchFalse, b)
  let
    uniq f = norm $ laPow freeCnt $ foldl Ap f uniqArgs
  easyArgs <- easy (churchTrue, uniq a) (churchFalse, uniq b)
  pure $ norm $ La $ foldl Ap (laPow freeCnt $ foldl Ap (Va freeCnt) uniqArgs) easyArgs

churchTrue = La $ La $ Va 1
churchFalse = La $ La $ Va 0
\end{code}

We can generalize the above algorithm to work on any number of distinct Bohm
trees. Some other time perhaps, as it entails descending several trees in
parallel and keeping track of new subsets on detecting differences.

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
<p><a onclick='hideshow("ui");'>&#9654; Toggle UI</a></p>
<div id='ui' style='display:none'>
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

\begin{code}
foreign export ccall "go" main
main = do
  aVal <- getContents
  nextInput
  bVal <- getContents
  let
    eab = do
      a <- debruijn [] =<< parse "A" aVal
      b <- debruijn [] =<< parse "B" bVal
      x <- distinguish a b
      pure $ foldr (.) id $ (. ("<br>\n"++)) <$>
        [ ("A = "++) . shows a
        , ("B = "++) . shows b
        , ("X = "++) . shows x
        , ("XA = "++) . shows (norm $ Ap x a)
        , ("XB = "++) . shows (norm $ Ap x b)
        ]

  putStr $ either ("error: "++) ($ "") eab
\end{code}

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
</div>
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

== See Also ==

I first encountered Böhm's Theorem in
https://www.youtube.com/watch?v=QVwm9jlBTik[a talk by David Turner on the
history of functional programming]. On one slide, he states "three really
important theorems about the lambda calculus". I had heard of the two
Church-Rosser theorems he cited, so I was surprised by my ignorance of the
third key result.

I was surprised again when I had trouble finding explanations online. Perhaps
computer scientists in general are less aware of Böhm's Theorem than they ought
to be. I came across Guerrini, Piperno, and Dezani-Ciancaglini,
https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.621.8250&rep=rep1&type=pdf['Why
Böhm's Theorem matters'], who make the theorem sound easy to implement. They
outline its proof without mentioning lambda calculus(!) and draw pretty
pictures with trees.

It inspired me to try coding Böhm's Theorem, but I soon found they had swept
some tricky problems under the rug. Luckily, their paper cited Gérard Huet's
https://hal.inria.fr/inria-00074664/document[implementation of Böhm's Theorem
in ML], and I also found other, more detailed proofs.

I eventually made it out alive and completed the above implementation. I
studied Huet's code to see how he had dealt with the corner cases that had
almost defeated me. I was shocked to find they weren't handled at all!
I thought there must be a bug. With Huet's help, I was able to clone:

 * https://gitlab.inria.fr/huet/cct/

(it needs the now-obsolete camlp4, so I built it with ocaml-3.12.1), and
verify that it handles the "hard hard case" just fine.

I realized Huet's code has no need to handle special cases because it deals
with one variable at a time, rather than add a bunch of them in one go, as we
do. His approach is simple and elegant.

Cleverer still, in each Bohm tree node, variables are numbered starting from 0
but in reverse order to De Bruijn indices. (They are also tagged with a second
number indicating the level of the tree they belong.) Thus eta expansion is
free, while our code must jump through some hoops to renumber variables when
eta-expanding.

I also learned from Huet that Böhm himself never published his theorem, except
in a techincal report; Barendregt later presented a proof in a book.
Furthermore, the theorem is crucial because it means there are only two
interesting models of lambda-calculus, the extensional model $D_\infty$ and
the intensional model $P(\omega)$ ("the graph model").

By the way, Barendregt also wrote about https://arxiv.org/abs/1812.02243[gems
of Corrado Böhm]. If you liked this algorithm, then you'll love his other
ideas!
