= Outcoding UNIX geniuses =

Lack of
https://en.wikipedia.org/wiki/Parametric_polymorphism[parametric polymorphism]
catches a programmer in
https://en.wikipedia.org/wiki/Morton's_fork[Morton's fork]: between a rock and
a hard place. We're forced to duplicate code or cast types. (It's worse for
theoreticians, who have no choice but to duplicate code because type casting
breaks everything.)

So why don't all languages support this feature? Because it's tough to do:
https://golang.org/doc/faq#generics[the designers of the Go language, including
famed former Bell Labs researchers, have been stumped for years].

Happily, a little theory rescues us. We'll see how 'type inference',
or 'type reconstruction', leads to parametric polymorphism in an interpreter
for https://en.wikipedia.org/wiki/Programming_Computable_Functions[PCF
(Programming Computable Functions)], a simply typed lambda calculus with
the base type `Nat` with the constant `0` and extended with:

 - `pred`, `succ`: these functions have type `Nat -> Nat` and return the
   predecessor and successor of their input; evaluating `pred 0` anywhere in
   a term returns the `Err` term which represents this exception.

 - `ifz-then-else`: when given 0, an `ifz` expression evaluates to its `then`
   branch, otherwise it evaluates to its `else` branch.

 - `fix`: the fixpoint operator, allowing recursion (but breaking
   normalization).

For convenience, we parse all natural numbers as constants of type `Nat`.
We also provide an `undefined` keyword that throws an error.

Terms that avoid the fixpoint operator are normalizing. In spite of guaranteed
termination, this system is surprisingly expressive even without `fix`. We can
sort lists! 

[pass]
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
<script src="hm.js"></script>
<p><button id="evalB">Run</button>
<button id="resetB">Reset</button>
<button id="sortB">Sort</button>
</p>
<p><textarea style="border: solid 2px; border-color: #999999" id="input" rows="10" cols="80">
</textarea></p>
<p><textarea id="output" rows="8" cols="80" readonly></textarea></p>
<textarea id="resetP" hidden>
two = succ (succ 0)
three = succ two
add = fix (\f m n.ifz m then n else f (pred m) (succ n))
mul = fix (\f t m n.ifz m then t else f (add t n) (pred m) n) 0
add two three
mul three three
let id = \x.x in id succ (id three)  -- Let-polymorphism.
</textarea>
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Some presentations of PCF also add the base type `Bool` along with constants
`True`, `False` and replace `ifz` with `if` and `iszero`, which is similar to
link:simply.html[our last interpreter].

To be fair to Go: for full-blown generics, we need recursive types and type
operators to define, say, a binary tree containing values of any given type.
Even then parametric polymorphism is only half the problem. The other half is
ad hoc polymorphism, which Haskell researchers only neatly solved in the late
1980s with type classes. Practical Haskell compilers also need more trickery
for unboxing.

== Look Ma, No Types! ==

We implement Algorithm W, which returns the most general type of a given closed
term despite missing some or even all type information. The algorithm
succeeds if and only if the given expression is 'typable', that is, when
certain types are assigned to the bindings lacking type annotations, it is
well-typed.

For example, Algorithm W infers the following expressions:

------------------------------------------------------------------------------
pred (succ 0)
\x y z.x z(y z)
\a b.succ (a 0)
\f x:X.f(f x)
------------------------------------------------------------------------------

have the following types:

------------------------------------------------------------------------------
Nat
(_2 -> _4 -> _5) -> (_2 -> _4) -> _2 -> _5]
(Nat -> Nat) -> _1 -> Nat
(X -> X) -> X -> X
------------------------------------------------------------------------------

The only base type is `Nat`. Names such as `X` or `_2` are 'type variables',
and can be supplied by the programmer or generated on demand. Then the inferred
type is most general, or 'principal', in the sense that:

  1. Substituting types such as `Nat` or `(Nat -> Nat) -> Nat`
  (sometimes called 'type constants' for clarity)
  for all the type variables results in a well-typed closed term.
  For example, in the last function above, instantiating `X` with `Nat`
  results in `\f:Nat -> Nat x:Nat.f(f x)` of type `Nat`.

  2. There are no other ways of typing the given expression.

== Let there be let ==

We generalize let expressions. So far, we have only allowed them at the top
level. We now allow `let _ = _ in _` anywhere we expect a term. For example:

  \x y.let z = \a b.a in z x y

Evaluating them is trivial:

  eval env (Let x y z) = eval env $ beta (x, y) z

That is, we simply add a new binding to the environment before evaluating the
let body. An easy exercise is to add this to our previous demos: after
trivially modify parsing and type-checking, it should just work.

But how should type inference interact with let?

A profitable strategy is to pay respect to the equals sign: we stipulate the
left and right sides of `(=)` are interchangeable, so the following program:

------------------------------------------------------------------------------
id = \x.x
id succ(id 0)
------------------------------------------------------------------------------

should expand to:

------------------------------------------------------------------------------
((\x.x) succ)((\x.x) 0)
------------------------------------------------------------------------------

which evaluates to 1.

In other words, `let` should behave as macros do in many popular languages. In
`id succ`, the symbol `id` should mean `\x:Nat -> Nat.x`, while in `id 0`, it
should mean `\x:Nat.x`.

This is an example of 'let-polymorphism', and its benefits are most apparent
when we consider invalid programs such as the following:

------------------------------------------------------------------------------
(\f.f succ(f 0)) (\x.x)
------------------------------------------------------------------------------

Without let-polymorphism, even with type inference we would be forced to
duplicate code to correct the above:

------------------------------------------------------------------------------
(\f g.f succ(g 0)) (\x.x) (\x.x)
------------------------------------------------------------------------------

(http://homepages.inf.ed.ac.uk/gdp/publications/LCF.pdf[The original PCF]
lacked type inference and hence let-polymorphism.)

== Memoized type inference ==

Our type inference algorithm could also treat `let` as a macro: we could
fully expand all let definitions before type checking.
However, expansion causes work to be repeated.

In the above example, we would determine the first `(\x.x)` has type `_0 -> _0`
where `_0` is a generated type variable, before deducing further that `_0` must
be `Nat -> Nat`. Afterwards, we would repeat computations to determine that the
second `(\x.x)` has type `_1 -> _1`, before deducing `_1` must be `Nat`.

In general, functions can be more complicated than `\x.x` and let expansions
can be deeply nested, leading to prohibitively many repeated computations.

The solution is to https://en.wikipedia.org/wiki/Memoization[memoize: cache the
results of a computation for later reuse]. We introduce 'generalized type
variables' for this purpose: a generalized type variable is replaced with a
fresh type variable on demand.

In our example above, we first use type inference to determine `id` has type `X
-> X` where `X` is a type variable. Next, we mark `X` as a generalized type
variable. Then each time `id` is used in an expression, we replace `X` with a
newly generated type variable before proceeding with type inference.

== Formalizing macros ==

Memoization is also useful for understanding the theory. Rather than
vaguely say `id` is a sort of macro, we say that `id = \x.x` has type `∀X.X ->
X`.  The symbol `∀` indicates a given type variable is generalized.  Lambda
calculus with generalized type variables from let-polymorphism is known as the
'Hindley-Milner type system', or HM for short. Like simply typed lambda
calculus, HM is strongly normalizing.

We might then wonder if this `∀` notation is redundant. Since let definition
are like macros, shouldn't we generalize all type variables returned by the
type inference algorithm? Why would we ever need to distinguish between
generalized type variables and plain type variables if they're always going
to be generalized?

The reason becomes clear when we consider lower-level let expressions.
Our code must mix generalized and ordinary type variables, and carefully keep
track of them in order to correctly infer types. Consider the following example
from Benjamin C. Pierce, ``Types and Programming Languages'',
where the language has base types `Nat` and `Bool`:

------------------------------------------------------------------------------
(\f:X->X x:X. let g=f in g 0)
  (\x:Bool. if x then True else False)
  True;
------------------------------------------------------------------------------

This program is invalid. But if we blithely assume all type variables in
let expressions should be generalized, then we would mistakenly conclude
otherwise. We would infer `g` has type `∀X.X->X`. In `g 0`, this would
generate a new type variable (that we then infer should be `Nat`).

Instead, we must infer `g` has type `X->X`, that is, `X` is an plain type
variable and not generalized. This enables type inference to find
two contradictory constraints (`X = Nat` and `X = Bool`) and reject the term.

On the other hand, we should generalize type variables in let expressions
absent from higher levels. For example, in the following expression:

------------------------------------------------------------------------------
\f:X->X x:X. let g=\y.f in g
------------------------------------------------------------------------------

type inference should determine the function `g` has type
`∀Y.Y->(X->X)->(X->X)`, that is, `Y` is generalized while `X` is not.

These details only matter when implementing languages. Users can blissfully
ignore the distinction, because in top-level let definitions, all type
variables are generalized, and in evaluated terms, all generalized type
variables are replaced by plain type variables. When else does a user ask
for a term's type?

Indeed, our demo will follow Haskell and omit the `(∀)` symbol. We'll say, for
example, the `const` function has type `a -> b -> a` even though `a` and `b`
are generalized type variables; its type is really `∀a b.a -> b ->a`.

== Halfway to Haskell ==

Syntax aside, we're surprisingly close to Haskell 98, which is based on HM
extended with the fixpoint operator. We lack many base types and primitive
functions, and we have if expressions instead of the nicer case expressions,
but these have little theoretical significance.

The juicy missing pieces are algebraic data types and type classes.

Later versions of Haskell go beyond Hindley-Milner to a variant of System F.
As a result, type inference is no longer guaranteed to succeed, and often the
programmer must supply annotations to help the type checker.

We would be close to ML if we had chosen eager evaluation instead of lazy
evaluation.

== Definitions ==

Despite the advanced capabilities of HM, we can almost reuse the data
structures of simply typed lambda calculus.

To our data type representing types, we add type variables and generalized type
variables: our `TV` and `GV` constructors. And to our data type representing
terms, we add a `Let` constructor to represent let expressions.

To keep the code simple, we show generalized type variables in a nonstandard
manner: we simply prepend an at sign to the variable name. It's understood
that `(@x -> y) -> @z` really means `∀@x @z.(@x -> y) -> @z`. Since we follow
Haskell's convention by showing non-generalized type variables for top-level
let expressions, under normal operation we'll never show a generalized type
variable. One would only show up if we, say, added a logging statement for
debugging.

\begin{code}
{-# LANGUAGE CPP #-}
#ifdef __HASTE__
import Haste.DOM
import Haste.Events
#else
import System.Console.Readline
#endif
import Control.Arrow
import Control.Monad
import Data.Char
import Data.List
import Data.Maybe
import Text.ParserCombinators.Parsec

data Type = Nat | TV String | GV String | Type :-> Type deriving Eq
data Term = Var String | App Term Term | Lam (String, Type) Term
  | Ifz Term Term Term | Let String Term Term | Err

instance Show Type where
  show Nat = "Nat"
  show (TV s) = s
  show (GV s) = '@':s
  show (t :-> u) = showL t ++ " -> " ++ show u where
    showL (_ :-> _) = "(" ++ show t ++ ")"
    showL _         = show t

instance Show Term where
  show (Lam (x, t) y)    = "\0955" ++ x ++ ":" ++ show t ++ showB y where
    showB (Lam (x, t) y) = " " ++ x ++ ":" ++ show t ++ showB y
    showB expr           = "." ++ show expr
  show (Var s)    = s
  show (App x y)  = showL x ++ showR y where
    showL (Lam _ _) = "(" ++ show x ++ ")"
    showL _         = show x
    showR (Var s)   = ' ':s
    showR _         = "(" ++ show y ++ ")"
  show (Ifz x y z) =
    "ifz " ++ show x ++ " then " ++ show y ++ " else " ++ show z
  show (Let x y z) =
    "let " ++ x ++ " = " ++ show y ++ " in " ++ show z
  show Err         = "*exception*"
\end{code}

== Parsing ==

The biggest change is the parsing of types in lambda abstractions. If omitted,
we supply the type variable `_` which indicates we should automatically
generate a unique variable name for it later. Any name but `Nat` is a
user-supplied type variable name.

We also rename `Let` to `TopLet` (for top-level let expressions) to avoid
clashing with our above `Let` constructor.

\begin{code}
data PCFLine = Empty | TopLet String Term | Run Term

line :: Parser PCFLine
line = (((eof >>) . pure) =<<) . (ws >>) $ option Empty $ do
  t <- term
  option (Run t) $ str "=" >> TopLet (getV t) <$> term where
  getV (Var s) = s
  term = ifz <|> letx <|> lam <|> app
  letx = Let <$> (str "let" >> v) <*> (str "=" >> term)
    <*> (str "in" >> term)
  ifz = Ifz <$> (str "ifz" >> term) <*> (str "then" >> term)
    <*> (str "else" >> term)
  lam = flip (foldr Lam) <$> between lam0 lam1 (many1 vt) <*> term where
    lam0 = str "\\" <|> str "\0955"
    lam1 = str "."
    vt   = (,) <$> v <*> option (TV "_") (str ":" >> typ)
  typ = ((str "Nat" >> pure Nat) <|> (TV <$> v)
    <|> between (str "(") (str ")") typ)
      `chainr1` (str "->" >> pure (:->))
  app = foldl1' App <$> many1 ((Var <$> v) <|> between (str "(") (str ")") term)
  v   = try $ do
    s <- many1 alphaNum
    when (s `elem` words "ifz then else let in") $ fail "unexpected keyword"
    ws
    pure s
  str = try . (>> ws) . string
  ws = spaces >> optional (try $ string "--" >> many anyChar)
\end{code}

== Type Inference ==

Type inference has two stages:

  1. We walk through the given closed term and record constraints as we go.
  Each constraint equates one type expression with another, for example,
  `X -> Y = Nat -> Nat -> Z`. We may introduce more type variables
  during this stage. We return a constraint set as well as a type expression
  representing the type of the closed term. At this point, the most general
  form of this type expression is unknown; in fact, it is unknown if the
  type expression even has a valid solution satisfying all the constraints.

  2. We walk through the set of constraints to find type substitutions for
  each type variable. We may introduce additional constraints during this
  stage, but in such a way that the process is guaranteed to terminate.
  By the end we know whether the given closed term can be typed. By
  applying all the type substitutions to the type expression of the closed
  term, we find its principal type.

In the first stage, the `gather` function recursively creates a constraint set
which we represent with a list of pairs; each pair consists of type expressions
which must be equal. We thread an integer throughout so we can easily generate
a new variable name different to all other variables. Our generated variables
are simply the next free integer prepended by an underscore. Users are
prohibited by the grammar from using underscores in their type variable names.

A variable whose name is anything but one of `fix pred succ 0` must
either be the bound variable in a lambda abstraction, or the left-hand side
of an equation in a let expression. Either way, its type is given in the
association list `gamma`. We call `instantiate` to generate fresh type
variables for any generalized type variables before returning.

If the variable name is absent from `gamma`, then the term is unclosed, which
is an error. We abuse the `GV` constructor to represent this error.

We're careful when handling a let expression: we only generalize those type
variables that are absent from `gamma` before recursively calling `gather`.

\begin{code}
readInteger s = listToMaybe $ fst <$> (reads s :: [(Integer, String)])

gather gamma i term = case term of
  Var "undefined" -> (TV $ '_':show i, [], i + 1)
  Var "fix" -> ((a :-> a) :-> a, [], i + 1) where a = TV $ '_':show i
  Var "pred" -> (Nat :-> Nat, [], i)
  Var "succ" -> (Nat :-> Nat, [], i)
  Var s
    | Just _ <- readInteger s -> (Nat, [], i)
    | Just t <- lookup s gamma ->
      let (t', _, j) = instantiate t i in (t', [], j)
    | otherwise -> (TV "_", [(GV $ "undefined: " ++ s, GV "?")], i)
  Lam (s, TV "_") u -> (x :-> tu, cs, j) where
    (tu, cs, j) = gather ((s, x):gamma) (i + 1) u
    x = TV $ '_':show i
  Lam (s, t) u -> (t :-> tu, cs, j) where
    (tu, cs, j) = gather ((s, t):gamma) i u
  App t u -> (x, [(tt, uu :-> x)] `union` cs1 `union` cs2, k + 1) where
    (tt, cs1, j) = gather gamma i t
    (uu, cs2, k) = gather gamma j u
    x = TV $ '_':show k
  Ifz s t u -> (tt, foldl1' union [[(ts, Nat), (tt, tu)], cs1, cs2, cs3], l)
    where (ts, cs1, j) = gather gamma i s
          (tt, cs2, k) = gather gamma j t
          (tu, cs3, l) = gather gamma k u
  Let s t u -> (tu, cs1 `union` cs2, k) where
    gen = generalize (concatMap (freeTV . snd) gamma) tt
    (tt, cs1, j) = gather gamma i t
    (tu, cs2, k) = gather ((s, gen):gamma) j u

instantiate ty i = f ty [] i where
  f ty m i = case ty of
    GV s | Just t <- lookup s m -> (t, m, i)
         | otherwise            -> (x, (s, x):m, i + 1) where
           x = TV ('_':show i)
    t :-> u -> (t' :-> u', m'', i'') where
      (t', m' , i')  = f t m  i
      (u', m'', i'') = f u m' i'
    _       -> (ty, m, i)

generalize fvs ty = case ty of
  TV s | s `notElem` fvs -> GV s
  s :-> t                -> generalize fvs s :-> generalize fvs t
  _                      -> ty

freeTV (a :-> b) = freeTV a ++ freeTV b
freeTV (TV tv)   = [tv]
freeTV _         = []
\end{code}

In the second stage, the function `unify` takes each constraint in turn,
and applies type substitutions that all seem obvious:

 1. If there are no constraints left, then we have successfully inferred the
 type.

 2. If both sides of a constraint are the same type expression, there is
 nothing to do.

 3. If one side is a type variable `X`, and it also appears somewhere on the
 other side, then we are attempting to create an infinite type, which is
 forbidden. Otherwise the constraint is something like `X = Y -> (Nat -> Y)`,
 and we substitute all occurences of `X` in the constraint set with the type
 expression on the other side.

 4. If both sides have the form `s -> t` for some type expressions `s` and `t`,
 then add two new constraints to the set: one equating the type expressions
 before the `->` type constructor, and the other equating those after.

 5. It turns out if none of the above conditions are satisfied, then the
 given term is invalid.

\begin{code}
unify ((GV s, GV "?"):_)   = Left s
unify []                   = Right []
unify ((s, t):cs) | s == t = unify cs
unify ((TV x, t):cs)
  | x `elem` freeTV t = Left $ "infinite: " ++ x ++ " = " ++ show t
  | otherwise         = ((x, t):) <$> unify (join (***) (subst (x, t)) <$> cs)
unify ((s, TV y):cs)
  | y `elem` freeTV s = Left $ "infinite: " ++ y ++ " = " ++ show s
  | otherwise         = ((y, s):) <$> unify (join (***) (subst (y, s)) <$> cs)
unify ((s1 :-> s2, t1 :-> t2):cs) = unify $ (s1, t1):(s2, t2):cs
unify ((s, t):_)      = Left $ "mismatch: " ++ show s ++ " /= " ++ show t

subst (x, t) ty = case ty of
  a :-> b       -> subst (x, t) a :-> subst (x, t) b
  TV y | x == y -> t
  _             -> ty
\end{code}

The function `typeOf` is little more than a wrapper around `gather` and `unify`.
It applies all the substitutions found during `unify` to the type expression
returned by `gather` to compute the principal type of a given closed term
in a given context.

\begin{code}
typeOf gamma term = foldl' (flip subst) ty <$> unify cs where
  (ty, cs, _) = gather gamma 0 term
\end{code}

== Evaluation ==

Evaluation is elementary compared to type inference. Once we're certain a
closed term is well-typed, we can ignore the types and evaluate as we would in
untyped lambda calculus.

If we only wanted the weak head normal form, then we could take shortcuts: we
could assume the first argument to any `ifz`, `pred`, or `succ` is a natural
number. However, we want the normal form, necessitating extra checks.

\begin{code}
eval env (Var "undefined") = Err
eval env t@(Ifz x y z) = case eval env x of
  Err -> Err
  Var s -> case readInteger s of
    Just 0 -> eval env y
    Just _ -> eval env z
    _      -> t
  _     -> t
eval env (Let x y z) = eval env $ beta (x, y) z
eval env (App m a) = let m' = eval env m in case m' of
  Err -> Err
  Lam (v, _) f -> eval env $ beta (v, a) f
  Var "pred" -> case eval env a of
    Err -> Err
    Var s -> case readInteger s of
      Just 0 -> Err
      Just i -> Var (show $ read s - 1)
      _      -> App m' (Var s)
    t -> App m' t
  Var "succ" -> case eval env a of
    Err -> Err
    Var s -> case readInteger s of
      Just i -> Var (show $ read s + 1)
      _      -> App m' (Var s)
    t -> App m' t
  Var "fix" -> eval env (App a (App m a))
  _ -> App m' a
eval env (Var v) | Just x  <- lookup v env = eval env x
eval _   term                              = term

beta (v, a) f = case f of
  Var s | s == v         -> a
        | otherwise      -> Var s
  Lam (s, t) m
        | s == v         -> Lam (s, t) m
        | s `elem` fvs   -> let s1 = newName s fvs in
          Lam (s1, t) $ rec $ rename s s1 m
        | otherwise      -> Lam (s, t) (rec m)
  App m n                -> App (rec m) (rec n)
  Ifz x y z              -> Ifz (rec x) (rec y) (rec z)
  Let x y z              -> Let x (rec y) (rec z)
  where
    rec = beta (v, a)
    fvs = fv [] a

fv vs (Var s) | s `elem` vs = []
              | otherwise   = [s]
fv vs (Lam (s, _) f)        = fv (s:vs) f
fv vs (App x y)             = fv vs x `union` fv vs y
fv vs (Let _ x y)           = fv vs x `union` fv vs y
fv vs (Ifz x y z)           = fv vs x `union` fv vs y `union` fv vs z

newName x ys = head $ filter (`notElem` ys) $ (s ++) . show <$> [1..] where
  s = dropWhileEnd isDigit x

rename x x1 term = case term of
  Var s | s == x    -> Var x1
        | otherwise -> term
  Lam (s, t) b
        | s == x    -> term
        | otherwise -> Lam (s, t) (rec b)
  App a b           -> App (rec a) (rec b)
  Ifz a b c         -> Ifz (rec a) (rec b) (rec c)
  Let a b c         -> Let a (rec b) (rec c)
  where rec = rename x x1
\end{code}

\begin{code}
norm env term = case eval env term of
  Err          -> Err
  Var v        -> Var v
  Lam (v, t) m -> Lam (v, t) (rec m)
  App m n      -> App (rec m) (rec n)
  Ifz x y z    -> Ifz (rec x) (rec y) (rec z)
  where rec = norm env
\end{code}

== User Interface ==

This is slightly different from our previous demo because our typing algorithm
returns a hopefully helpful message instead of `Nothing` on error.

\begin{code}
#ifdef __HASTE__
main = withElems ["input", "output", "evalB", "resetB", "resetP",
    "sortB", "sortP"] $
  \[iEl, oEl, evalB, resetB, resetP, sortB, sortP] -> do
  let
    reset = getProp resetP "value" >>= setProp iEl "value"
      >> setProp oEl "value" ""
    run (out, env) (Left err) =
      (out ++ "parse error: " ++ show err ++ "\n", env)
    run (out, env@(gamma, lets)) (Right m) = case m of
      Empty      -> (out, env)
      Run term   -> case typeOf gamma term of
        Left m  -> (concat
           [out, "type error: ", show term, ": ", m, "\n"], env)
        Right t -> (out ++ show (norm lets term) ++ "\n", env)
      TopLet s term -> case typeOf gamma term of
        Left m  -> (concat
           [out, "type error: ", show term, ": ", m, "\n"], env)
        Right t  -> (out ++ "[" ++ s ++ ":" ++ show t ++ "]\n",
          ((s, generalize [] t):gamma, (s, term):lets))
  reset
  resetB `onEvent` Click $ const reset
  sortB `onEvent` Click $ const $
    getProp sortP "value" >>= setProp iEl "value" >> setProp oEl "value" ""
  evalB `onEvent` Click $ const $ do
    es <- map (parse line "") . lines <$> getProp iEl "value"
    setProp oEl "value" $ fst $ foldl' run ("", ([], [])) es
#else
repl env@(gamma, lets) = do
  let redo = repl env
  ms <- readline "> "
  case ms of
    Nothing -> putStrLn ""
    Just s  -> do
      addHistory s
      case parse line "" s of
        Left err  -> do
          putStrLn $ "parse error: " ++ show err
          redo
        Right Empty -> redo
        Right (Run term) -> do
          case typeOf gamma term of
            Left msg -> putStrLn $ "bad type: " ++ msg
            Right t  -> do
              putStrLn $ "[" ++ show t ++ "]"
              print $ norm lets term
          redo
        Right (TopLet s term) -> case typeOf gamma term of
          Left msg -> putStrLn ("bad type: " ++ msg) >> redo
          Right t  -> do
            putStrLn $ "[" ++ s ++ " : " ++ show t ++ "]"
            repl ((s, generalize [] t):gamma, (s, term):lets)

main = repl ([], [])
#endif
\end{code}

== The world's simplest list API ==

What's the desert island function from Haskell's `Data.List` package?

It's `foldr`. We can build everything else from right-folding over a list:

------------------------------------------------------------------------------
map f = foldr (\x xs -> f x : xs) []
head = foldr const undefined
null = foldr (const . const False) True
foldl = foldr . flip
------------------------------------------------------------------------------

The `tail` function is less elegant. We apply the same trick used in computing
the predecessor of a Church numeral:

------------------------------------------------------------------------------
tail = snd $ foldr (\x (as, _) -> (x:as, as)) ([], undefined)
------------------------------------------------------------------------------

Similarly, we can write a `foldr`-based function that inserts an element into a
sorted list so it remains sorted:

------------------------------------------------------------------------------
ins y xs = case foldr f ([y], []) xs of ([],  t) -> t
                                        ([h], t) -> h:t

f x ([] , rest)             = ([] , x:rest)
f x ([y], rest) | x < y     = ([] , x:y:rest)
                | otherwise = ([y], x:rest)
------------------------------------------------------------------------------

So insertion sort can be written:

------------------------------------------------------------------------------
sort :: Ord a => [a] -> [a]
sort = foldr ins []
------------------------------------------------------------------------------

Our type system turns out to accommodate
https://en.wikipedia.org/wiki/Church_encoding#Represent_the_list_using_right_fold[lists
represented with right fold], which may be easier to understand in Haskell:

------------------------------------------------------------------------------
nil = \c n->n
con = \h t c n->c h(t c n)
example = con 3(con 1(con 4 nil))
example (:) []  -- [3, 1, 4]
------------------------------------------------------------------------------

By translating the above to lambda calculus, we obtain a sorting function
without `fix`. (We do use `fix` in our less-than function, but in a practical
language this would be a built-in primitive.)

It almost seems we're cheating because we're piggybacking off the representation
of the list to carry out a form of recursion. More generally, functional
representations of data sometimes possess this trait: it can seem ridiculously
simple to express complex tasks.

[pass]
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
<textarea id="sortP" hidden>
-- Insertion sort without fix. Slow.
pair=\x y f.f x y
fst=\p.p(\x y.x)
snd=\p.p(\x y.y)
nil=\c n.n
cons=\h t c n.c h(t c n)
null=\l.l(\h t.0)1
head=\l.l(\h t.h)undefined
-- This fix doesn't count; less-than is usually a primitive built-in.
lt=fix(\f x y.ifz y then 0 else ifz x then 1 else f (pred x) (pred y))
f=\x p.ifz null (fst p) then ifz lt x (head (fst p)) then pair (fst p) (cons x (snd p)) else pair nil (cons x (cons (head (fst p)) (snd p))) else pair nil (cons x (snd p))
ins=\x l.let q = l f (pair (cons x nil) nil) in (ifz null (fst q) then (cons (head (fst q)) (snd q)) else snd q)
sort=\l.l ins nil
sort (cons 3(cons 1(cons 4(cons 1(cons 5 nil)))))
</textarea>
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
