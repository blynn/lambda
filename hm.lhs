= Outcoding UNIX geniuses =

Static types catch bugs at compile time, preventing costly incidents.
Unfortunately, in many languages, supplying type information is so laborious
and stifling that some simply give up. Furthermore, some languages lack
https://en.wikipedia.org/wiki/Parametric_polymorphism[parametric polymorphism],
forcing a programmer to choose between duplicating code or type casting.

Advances in theory solve these problems, though mainstream programmers are
unaware:

 * Popular authors Bruce Eckel and Robert C. Martin seem to mistakenly believe
 https://docs.google.com/document/d/1aXs1tpwzPjW9MdsG5dI7clNFyYayFBkcXwRDo-qvbIk/preview[strong typing implies verbosity], and worse still, http://blog.cleancoder.com/uncle-bob/2017/01/11/TheDarkPath.html[testing conquers all].
 Tests are undoubtedly invaluable, but at best they
 https://en.wikipedia.org/wiki/Proof_by_example[``prove'' by example]. As in
 mathematics, the one true path lies in rigorous proofs of correctness. That
 is, we need strong static types so that logic can work its magic. One could
 even argue a test-heavy approach helps attackers find exploits: the test
 cases you choose may hint at the bugs you overlooked.

 * https://golang.org/doc/faq#generics[The designers of the Go language,
 including famed former Bell Labs researchers, have been stumped by
 polymorphism for years].

Why is this so? Perhaps people think the theory is arcane, dry, and
impractical?

By working through some programming interview questions, we'll find the
relevant theory is surprisingly accessible. We quickly arrive at a simple 'type
inference' or 'type reconstruction' algorithm that seems too good to be true:
it powers strongly typed languages that support parametric polymorphism
without requiring any type declarations.

[pass]
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
<script src="hm.js"></script>
<p><textarea style="border: solid 2px; border-color: #999999" id="input" rows="10" cols="80">
V "length"
V "length" :@ S "hello"
Lam "x" (V "x" :@ I 2)
Lam "x" ((V "+" :@ V "x") :@ I 42)
Lam "x" (V "+" :@ (V "x" :@ I 42))
Lam "x" (V "x")
Lam "x" (Lam "y" (V "x"))
Lam "x" (Lam "y" (Lam "z" ((V "x" :@ V "z") :@ (V "y" :@ V "z"))))
</textarea></p>
<p><button id="inferB">Infer!</button>
</p>
<p><textarea id="output" rows="10" cols="80" readonly></textarea></p>
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The above type inference demo is a bit ugly; link:pcf.html[our next interpreter
will have a parser for a nice input language] but for now we'll make do
without one. The default value of the input text area describes the abstract
syntax trees of:

------------------------------------------------------------------------------
length
length "hello"
\x -> x 2
\x -> (+) x 42
\x -> (+) (x 42)
\x -> x
\x y -> x
\x y z -> x z(y z)
------------------------------------------------------------------------------

Clicking the button infers their types:

------------------------------------------------------------------------------
String -> Int
Int
Int -> a -> a
Int -> Int
(Int -> Int) -> Int -> Int
a -> a
a -> b -> a
(a -> b -> c) -> (a -> b) -> a -> c
------------------------------------------------------------------------------

Only the `length` and `(+)` functions have predefined types. An algorithm
figures out the rest.

Before presenting the questions, let's get some paperwork out of the way:

\begin{code}
{-# LANGUAGE CPP #-}
{-# LANGUAGE PackageImports #-}
#ifdef __HASTE__
import Haste.DOM
import Haste.Events
#endif
import Text.ParserCombinators.Parsec hiding (State)
import Text.Read hiding (get)
import Control.Arrow
import Control.Monad
import "mtl" Control.Monad.State  -- Haste has 2 versions of the State Monad.
\end{code}

Lastly, to be fair to Go: for full-blown generics, we need algebraic data types
and link:typo.html[type operators] to define, say, a binary tree containing
values of any given type. Even then, parametric polymorphism is only half the
problem. The other half is ad hoc polymorphism, which Haskell researchers only
neatly solved in the late 1980s with type classes. Practical Haskell compilers
also need more type trickery for unboxing.

== 1. Identifying twins ==

Determine if two binary trees of integers are equal.

'Solution:' I'd love to be asked this question so I could give the two-word
answer `deriving Eq`:

------------------------------------------------------------------------------
data Tree a = Leaf a | Branch (Tree a) (Tree a) deriving Eq
------------------------------------------------------------------------------

https://www.haskell.org/onlinereport/derived.html[Haskell's derived instance
feature] automatically works on any algebraic data type built on any type for
which equality makes sense. It even works for mutually recursive data types
(https://hackage.haskell.org/package/containers/docs/Data-Tree.html[see
`Data.Tree`]):

------------------------------------------------------------------------------
data Tree a = Node a (Forest a) deriving Eq
data Forest a = Forest [Tree a] deriving Eq
------------------------------------------------------------------------------

Perhaps my interviewer would ask me to explain `deriving Eq` does.
Roughly speaking, it generates code like the following, saving the
programmer from stating the obvious:

------------------------------------------------------------------------------
data Tree a = Leaf a | Branch (Tree a) (Tree a)

eq (Leaf x)       (Leaf y)       = x == y
eq (Branch xl xr) (Branch yl yr) = eq xl yl && eq xr yr
eq _              _              = False
------------------------------------------------------------------------------

== 2. On assignment ==

This time, one of the trees may contain variables in place of integers.
Can we assign integers to all variables so the trees are equal?
The same variable may appear more than once.

'Solution:' We extend our data structure to hold variables:

------------------------------------------------------------------------------
data Tree a = Var String | Leaf a | Branch (Tree a) (Tree a)
------------------------------------------------------------------------------

As before, we traverse both trees and look for nodes that differ in value
or type. If one is a variable, then we record a constraint, that is, a variable
assignment that is required for the trees to be equal such as `a = 4` or `b =
2`.  If there are conflicting values for the same variable, then we indicate
failure by returning `Nothing`. Otherwise we return `Just` the list of
assignments found.

------------------------------------------------------------------------------
solve (Leaf x)       (Leaf y)       as | x == y = Just as
solve (Var v)        (Leaf x)       as = addConstraint v x as
solve l@(Leaf _)     r@(Var _)      as = solve r l as
solve (Branch xl xr) (Branch yl yr) as = solve xl yl as >>= solve xr yr
solve _              _              _  = Nothing

addConstraint v x cs = case lookup v cs of
  Nothing           -> Just $ (v, x):cs
  Just x' | x == x' -> Just cs
  _                 -> Nothing
------------------------------------------------------------------------------

== 3. Both Sides, Now ==

Now suppose leaf nodes in both trees can hold integer variables. Can two
trees be made equal by assigning certain integers to the variables? If so, find
the most general solution.

'Solution:' We proceed as before, but now we may encounter constraints such as
`a = b`, which equate two variables. To handle such a constraint, we pick one
of the variables, such as `a`, and replace all occurrences of `a` with the
other side, which is `b` in our example. This eliminates `a` from all
constraints. Eventually, all our constraints have an integer on at least one
side, which we check for consistency.

We discard redundant constraints where the same variable appears on both sides,
such as `a = a`. Thus a variable may wind up with no integer assigned to it,
which means if a solution exists, it can take any value.

For clarity, we separate the gathering of constraints from their unification.
Lazy evaluation means these steps are actually interleaved, but our code will
appear to solve the problem in two phases.

Also for clarity, our code is inefficient: it's likely faster to maintain a
`Data.Map` of substitutions, have each new substitution affect this map, and
only apply the substitution at the last minute.

------------------------------------------------------------------------------
data Tree a = Var String | Leaf a | Branch (Tree a) (Tree a) deriving Show

gather (Leaf x) (Leaf y) | x == y    = Just []
gather (Branch xl xr) (Branch yl yr) = (++) <$> gather xl yl <*> gather xr yr 
gather (Var _) (Branch _ _)          = Nothing
gather x@(Var v) y                   = Just [(x, y)]
gather x y@(Var _)                   = gather y x
gather _ _                           = Nothing

unify acc [] = Just acc
unify acc ((Leaf a, Leaf a')  :rest) | a == a' = unify acc rest
unify acc ((Var x , Var x')   :rest) | x == x' = unify acc rest
unify acc ((Var x , t)        :rest)           = unify ((x, t):acc) $
  join (***) (sub x t) <$> rest
unify acc ((t     , v@(Var _)):rest)           = unify acc ((v, t):rest)
unify _   _                                    = Nothing

sub x t a = case a of Var x' | x == x' -> t
                      Branch l r       -> Branch (sub x t l) (sub x t r)
                      _                -> a

solve t u = unify [] =<< gather t u
------------------------------------------------------------------------------

The peppering of `acc` throughout the definition of `unify` is mildly 
irritating. We can remove a few with an explicit case statement (which is
what happens behind the scenes anyway):

------------------------------------------------------------------------------
unify acc a = case a of
  []                                   -> Just acc
  ((Leaf a, Leaf a')  :rest) | a == a' -> unify acc rest
  ((Var x , Var x')   :rest) | x == x' -> unify acc rest
  ((Var x , t)        :rest)           -> unify ((x, t):acc) $
    join (***) (sub x t) <$> rest
  ((t     , v@(Var _)):rest)           -> unify acc ((v, t):rest)
  _                                    -> Nothing
------------------------------------------------------------------------------

We'll soon see a more thorough way to clean the code.

== 4. Once more, with subtrees ==

What if variables can represent subtrees?

'Solution:' Although we've significantly generalized the problem, our answer 
almost remains the same.

We remove one case from the `gather` function, as it is now legal to equate a
variable to a subtree. Then we modfiy one case to the `unify` function: before
we perform a substitution, we first check that our variable only appears on one
side to avoid infinite recursion. Lastly, we add a case to `unify` when
both sides are branches.

We take this opportunity to define `unify` using the state monad, which saves
us from explicitly referring to the list of assignments found so far, that is,
the list previously known as `acc`. To a first approximation, we're employing
macros allows us to hide them.

\begin{code}
data Tree a = Var String | Leaf a | Branch (Tree a) (Tree a) deriving Show

treeSolve :: (Show a, Eq a) => Tree a -> Tree a -> Maybe [(String, Tree a)]
treeSolve t1 t2 = (`evalState` []) . unify =<< gather t1 t2 where
  gather (Branch xl xr) (Branch yl yr) = (++) <$> gather xl yl <*> gather xr yr 
  gather (Leaf x) (Leaf y) | x == y    = Just []
  gather v@(Var _) x                   = Just [(v, x)]
  gather t v@(Var _)                   = gather v t
  gather _ _                           = Nothing

  unify :: Eq a =>
    [(Tree a, Tree a)] -> State [(String, Tree a)] (Maybe [(String, Tree a)])
  unify []                                 = Just <$> get
  unify ((Branch a b, Branch a' b'):rest)  = unify $ (a, a'):(b, b'):rest
  unify ((Leaf a, Leaf a'):rest) | a == a' = unify rest
  unify ((Var x, Var x'):rest)   | x == x' = unify rest
  unify ((Var x, t):rest)                  = if twoSided t then pure Nothing
    else modify ((x, t):) >> unify (join (***) (sub x t) <$> rest) where
      twoSided (Branch l r)     = twoSided l || twoSided r
      twoSided (Var y) | x == y = True
      twoSided _                = False
  unify ((t, v@(Var _)):rest)              = unify $ (v, t):rest
  unify _                                  = pure Nothing

  sub x t a = case a of Var x' | x == x' -> t
                        Branch l r       -> Branch (sub x t l) (sub x t r)
                        _                -> a
\end{code}

Here's a demo of the above code:

[pass]
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
<script src="hm.js"></script>
<p><textarea style="border: solid 2px; border-color: #999999" id="treeIn" rows="10" cols="80">
((a a) d) (((3 5) (b c)) d)
</textarea></p>
<p><button id="treeB">Solve</button>
</p>
<p><textarea id="treeOut" rows="8" cols="80" readonly></textarea></p>
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The given example ought to be enough enough to understand the input format,
which is parsed by the following:

\begin{code}
treePair :: Parser (Tree Int, Tree Int)
treePair = do
  spaces
  t <- tree
  spaces
  u <- tree
  spaces
  eof
  pure (t, u)

tree :: Read a => Parser (Tree a)
tree = tr where
  tr = leaf <|> branch
  branch = between (char '(') (char ')') $ do
    spaces
    l <- tr
    spaces
    r <- tr
    spaces
    pure $ Branch l r
  leaf = do
    s <- many1 alphaNum
    pure $ case readMaybe s of
      Nothing -> Var s
      Just a -> Leaf a
\end{code}

== 5. Type inference! ==

Design a language based on lambda calculus where integers and strings are
primitve types, and where we can deduce whether a given expression is
'typable', that is, whether types can be assigned to the untyped bindings so
that the expression is well-typed. If so, find the most general type.

For example, the expression `\f -> f 2` which takes its first argument `f` and
applies to the integer 2 must have type `(Int -> u) -> u`. Here, `u` is a 'type
variable', that is, we can substitute `u` with any type.
This is known as 'parametric polymorphism'.

More precisely the inferred type is most general, or 'principal' if:

  1. Substituting types such as `Int` or `(Int -> Int) -> Int`
  (sometimes called 'type constants' for clarity)
  for all the type variables results in a well-typed closed term.

  2. There are no other ways of typing the given expression.

'Solution:' We define an abstract syntax tree for an expression in our
language: applications, lambda abstractions, variables, integers, and strings:

\begin{code}
infixl 5 :@
data Expr = Expr :@ Expr | Lam String Expr | V String | I Int | S String
  deriving Read
\end{code}

So far we have simultaneously traversed two trees to generate constraints.
This time, we traverse a single abstract syntax tree. The constraints we
generate along the way equate types, which are represented with another data
type:

\begin{code}
infixr 5 :->
data Type = T String | Type :-> Type | TV String deriving Show
\end{code}

The `T` constructor is for primitive data types, which are `Int` and `String`.
The `(:->`) constructor is for functions, and the `TV` constructor is for
constructing 'type variables'.

The rules for building constraints from expressions are what we might expect:

 * The type of a primitive value is its corresponding type; for example, 5 has
 type `T "Int"` and "Hello, World" has type `T "String"`.

 * For an application `f x`, we recursively determine the type `tf` of `f`
and `tx` of `x` (possibly gathering new constraints along the way), generate a
new type variable `tfx` to return and generate the constraint that `tf` is
`tx :-> tfx`.

 * For a lambda abstraction `\x.t`, we generate a new type variable `tx` to
represent the type of `x`. Then we recursively find the type `tt` of `t`
being careful to assign the type `tx` to any free occurrence of `x`, and
return the type `tx :-> tt` for the lambda.

Bookkeeping is fiddly. To guarantee a unique name for each type variable, we
maintain a counter which we increment for each new name. We also maintain
an environment `gamma` that records the types of variables in lambda
abstractions.

We want more than assignments satisfying the constraints: we also want the type
of the given expression. Accordingly, we modify `gather` to return the type of
an expression as well as the type constraints it requires.

\begin{code}
gather :: [(String, Type)] -> Expr -> State ([(Type, Type)], Int) Type
gather gamma expr = case expr of
  I _ -> pure $ T "Int"
  S _ -> pure $ T "String"
  f :@ x -> do
    tf <- gather gamma f
    tx <- gather gamma x
    tfx <- newTV
    (cs, i) <- get
    put ((tf, tx :-> tfx):cs, i)
    pure tfx
  V s -> let Just tv = lookup s gamma in pure tv
  Lam x t -> do
    tx <- newTV
    tt <- gather ((x, tx):gamma) t
    pure $ tx :-> tt
  where
    newTV = do
      (cs, i) <- get
      put (cs, i + 1)
      pure $ TV $ 't':show i
\end{code}

We employ the same unification strategy:

 1. If there are no constraints left, then we have successfully inferred the
 type.

 2. If both sides have the form `s -> t` for some type expressions `s` and `t`,
 then add two new constraints to the set: one equating the type expressions
 before the `->` type constructor, and the other equating those after.

 3. If both sides of a constraint are the same, then we simply move on.

 4. If one side is a type variable `t`, and `t` also appears somewhere on the
 other side, then we are attempting to create an infinite type, which is
 forbidden. Otherwise the constraint is something like `t = u -> (Int -> u)`,
 and we substitute all occurences of `t` in the constraint set with the type
 expression on the other side.

 5. If none of the above applies, then the given term is untypable.

\begin{code}
unify :: [(Type, Type)] -> State [(String, Type)] (Maybe [(String, Type)])
unify []                            = Just <$> get
unify ((tx :-> ty, ux :-> uy):rest) = unify $ (tx, ux):(ty, uy):rest
unify ((T t,  T u) :rest) | t == u  = unify rest
unify ((TV v, TV w):rest) | v == w  = unify rest
unify ((TV x, t)   :rest)           = if twoSided t then pure Nothing
  else modify ((x, t):) >> unify (join (***) (sub (x, t)) <$> rest)
  where
    twoSided (t :-> u)       = twoSided t || twoSided u
    twoSided (TV y) | x == y = True
    twoSided _               = False
unify ((t, v@(TV _)):rest) = unify ((v, t):rest)
unify _ = pure Nothing

sub (x, t) y = case y of
  TV x' | x == x' -> t
  a :-> b         -> sub (x, t) a :-> sub (x, t) b
  _               -> y

solve gamma x = foldr sub ty <$> evalState (unify cs) [] where
  (ty, (cs, _)) = runState (gather gamma x) ([], 0)
\end{code}

This algorithm is known as Algorithm W, and is the heart of the
link:pcf.html['Hindley-Milner type system'], or HM for short.

== Example ==

Let's walk through an example. The expression `\f -> f 2` would be represented
as the `Expr` tree `Lam "f" (V "f" :@ I 2)`. Calling `gather` on this tree
consists of the following:

  1. Generate a new type variable `t`.
  2. Recursively invoke `gather` on the lambda body to find its type `u`,
  with the local constraint that the symbol `f` has type `t`.
  3. Return the type `t -> u`.

Step 2 expands to the following:

  1. Recursively invoke `gather` on the left and right children of the `(:@)`
  node to find their types `a` and `b`.
  2. Generate a new type variable `c`.
  3. Add the global constraint that `a` has type `b -> c`.
  4. Return the type `c`.

In step 1, the left child is the symbol `f`, which has type `t` because of
the local constraint generated by the `Lam` case, while the right child has
type `TInt` because it is the integer constant 2. Neither child generates any
more constraints.

Unification combines these constraints to find `\f -> f 2` has type
`(Int -> u) -> u`.

== UI ==

We predefine type signatures of certain functions:

\begin{code}
prelude :: [(String, Type)]
prelude = [
  ("+", T "Int" :-> T "Int" :-> T "Int"),
  ("length", T "String" :-> T "Int")]
\end{code}

These become the initial environment in our demo:

\begin{code}
#ifdef __HASTE__
main = withElems ["treeIn", "treeOut", "treeB", "input", "output", "inferB"] $
  \[treeIn, treeOut, treeB, iEl, oEl, inferB] -> do
  treeB `onEvent` Click $ const $ do
    s <- getProp treeIn "value"
    case parse treePair "" s of
      Left (err) -> setProp treeOut "value" $ "parse error: " ++ show err
      Right (t, u) -> case treeSolve t u of
        Nothing -> setProp treeOut "value" "no solution"
        Just as -> setProp treeOut "value" $ unlines $ show <$> as
  inferB `onEvent` Click $ const $ do
    s <- getProp iEl "value"
    setProp oEl "value" $ unlines $ map (\xstr -> case readMaybe xstr of
      Nothing -> "READ ERROR"
      Just x -> maybe "BAD TYPE" show (solve prelude x)) $ lines s
#else
main = do
  print $ solve [] (Lam "x" (V "x" :@ I 2))
  print $ solve prelude (Lam "x" (V "+" :@ V "x" :@ I 42))
#endif
\end{code}
