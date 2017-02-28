= Type Inference =

https://en.wikipedia.org/wiki/Programming_Computable_Functions[PCF
(Programming Computable Functions)] is a simply typed lambda calculus with
the base type `Nat` with the constant `0` and extended with:

 - `pred`, `succ`: these functions have type `Nat -> Nat` and return the
   predecessor and successor of their input; we define `pred 0 = 0` so `pred`
   is total.

 - `ifz-then-else`: when given 0, an `ifz` expression evaluates to its `then`
   branch, otherwise it evaluates to its `else` branch.

 - `fix`: the fixpoint operator. This allows recursion, but breaks
   normalization.

[pass]
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
<script src="pcf.js"></script>
<p><button id="evalB">Run</button>
<button id="resetB">Reset</button>
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

== Look Ma, No Types! ==

To avoid essentially repeating our previous demo, we take this opportunity to
introduce 'type inference' or 'type reconstruction'. We implement Algorithm W,
which returns the most general type of a given closed term despite the lack of
some or even all type information. The algorithm fails if and only if the given
expression cannot be well-typed.

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

We also generalize let expressions. So far, we have only allowed them at the
top level. We now allow `let _ = _ in _` anywhere we expect a term. For example:

  \x y.let z = \a b.a in z x y

Evaluating them is trivial:

  eval env (Let x y z) = eval ((x, y):env) z

That is, we simply add a new binding to the environment before evaluating the
let body. An easy exercise is to add this to our previous demos: after
trivially modify parsing and type-checking, it should just work.

But it's less easy in the presence of type inference.
How should type inference interact with let?

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

== Efficient type inference ==

Our type inference algorithm could also treat `let` as a macro: we could
fully expand all let definitions before performing type checking.
However, expansion causes work to be repeated.

In the above example, we would first determine `(\x.x)` has type `_0 -> _0`
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
-> X` where `X` is a type variable. Next, we change `X` to a generalized type
variable. Then each time `id` is used in an expression, we replace `X` with a
newly generated type variable before proceeding with type inference.

== Formalizing macros ==

This optimization is also useful for understanding the theory. Rather than
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

Later versions of Haskell go beyond Hindley-Milner to a variant of a system
known as System F. As a result, type inference is no longer guaranteed to
succeed, and often the programmer must supply annotations to help the type
checker.

== Definitions ==

Despite the advanced capabilities of HM, we can almost reuse the data
structures of simply typed lambda calculus.

To our data type representing types, we add type variables and generalized type
variables: our `TV` and `GV` type constructors. And to our data type
representing terms, we add a `Let` type constructor to represent let
expressions.

To keep the code simple, we show generalized type variables in a nonstandard
manner: we simply prepend an asterisk to the variable name. It's understood
that `(*x -> y) -> *z` really means `∀*x *z.(*x -> y) -> *z`. Since we follow
Haskell's convention by showing non-generalized type variables for top-level
let expressions, under normal operation we'll never show a generalized type
variable. One would only show up if we, say, added a logging statement for
debugging

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
import Text.ParserCombinators.Parsec

data Type = Nat | TV String | GV String | Type :-> Type deriving Eq
data Term = Var String | App Term Term | Lam (String, Type) Term
  | Ifz Term Term Term | Let String Term Term

instance Show Type where
  show Nat = "Nat"
  show (TV s) = s
  show (GV s) = '*':s
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
\end{code}

== Parsing ==

The biggest change is the parsing of types in lambda abstractions. If omitted,
we supply the type variable `_` which indicates we should automatically
generate a unique variable name for it later. Any name but `Nat` is a
user-supplied type variable name.

We also rename `Let` to `TopLet` (for top-level let expressions) to avoid
clashing with our above `Let` type constructor.

\begin{code}
data PCFLine = Empty | TopLet String Term | Run Term

line :: Parser PCFLine
line = (((eof >>) . pure) =<<) . (ws >>) $ option Empty $ do
  t <- term
  option (Run t) $ str "=" >> TopLet (getV t) <$> term where
  getV (Var s) = s
  term = ifz <|> letx <|> lam <|> app
  letx = do
    str "let"
    lhs <- v
    str "="
    rhs <- term
    str "in"
    body <- term
    pure $ Let lhs rhs body
  ifz = do
    str "ifz"
    cond <- term
    str "then"
    bfalse <- term
    str "else"
    btrue <- term
    pure $ Ifz cond bfalse btrue
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
  during this stage. We return a constraint as well as a type expression
  representing the type of the closed term. At this point, the most general
  form of this type expression is unknown; in fact, it is unknown if the
  type expression even has a valid solution satisfying all the constraints.

  2. We walk through the set of constraints to find type substitutions for
  each type variable. We may introduce additional constraints during this
  stage, but in such a way that the process is guaranteed to terminate.
  By the end we know whether the given closed term can be typed, and in fact,
  by applying all the type substitutions we found to the type expression of the
  closed term, we find its principal type.

In the first stage, the `gather` function recursively creates a constraint set
which we represent with a list of pairs; each pair consists of type expressions
which must be equal. We thread an integer throughout so we can easily generate
a new variable name different to all other variables. Our generated variables
are simply the next free integer prepended by an underscore. Users are
prohibited by the grammar from using underscores in their type variable names.

A variable whose name is anything but one of ``fix, pred, succ, 0'' must
either be the bound variable in a lambda abstraction, or the left-hand side
of an equation in a let expression. Either way, its type is given in the
association list `gamma`. We call `instantiate` to generate fresh type
variables for any generalized type variables before returning.

If the variable name is absent from `gamma`, then the term is unclosed, which
is an error. We abuse the `GV` type constructor to represent this error.

We're careful when handling a let expression: we only generalize those type
variables that are absent from `gamma` before recursively calling `gather`.

\begin{code}
gather gamma i term = case term of
  Var "fix" -> ((a :-> a) :-> a, [], i + 1) where a = TV $ '_':show i
  Var "pred" -> (Nat :-> Nat, [], i)
  Var "succ" -> (Nat :-> Nat, [], i)
  Var "0" -> (Nat, [], i)
  Var s
    | Just t <- lookup s gamma ->
      let (t', _, j) = instantiate t i in (t', [], j)
    | otherwise -> (TV s, [(GV s, GV "?!?")], i)
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
  Let s t u
    | Right tt <- typeOf gamma t ->
      let gen = generalize (concatMap (freeTV . snd) gamma) tt
      in gather ((s, gen):gamma) i u where

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
unify []                   = Right []
unify ((s, t):cs) | s == t = unify cs
unify ((GV s, GV "?!?"):_) = Left $ "undefined: " ++ s
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
in a given context `gamma`.

\begin{code}
typeOf gamma term = foldl' (flip subst) ty <$> unify cs where
  (ty, cs, _) = gather gamma 0 term
\end{code}

== Evaluation ==

We almost have the same old evaluation function. Type inference is the tricky
part; once we're certain a closed term is well-typed, we can ignore the types
and evaluate as we would in untyped lambda calculus.

Thanks to theory, expressions not involving the fixpoint operator are
guaranteed to terminate.

\begin{code}
eval env (Ifz x y z) = eval env $ case eval env x of
  Var "0"  -> y
  _        -> z
eval env (Let x y z) = eval ((x, y):env) z
eval env (App m a) = let m' = eval env m in case m' of
  Lam (v, _) f -> let
    beta (Var s) | s == v         = a
                 | otherwise      = Var s
    beta (Lam (s, t) m)
                 | s == v         = Lam (s, t) m
                 | s `elem` fvs   = let s1 = newName s fvs in
                   Lam (s1, t) $ beta $ rename s s1 m
                 | otherwise      = Lam (s, t) (beta m)
    beta (App m n)                = App (beta m) (beta n)
    beta (Ifz x y z)              = Ifz (beta x) (beta y) (beta z)
    beta (Let x y z)              = Let x (beta y) (beta z)
    fvs = fv env [] a
    in eval env $ beta f
  Var "pred" -> case eval env a of
    Var "0"  -> Var "0"
    Var s    -> Var (show $ read s - 1)
  Var "succ" | Var s <- eval env a -> Var (show $ read s + 1)
  Var "fix" -> eval env (App a (App m a))
  _ -> App m' a 

eval env term@(Var v) | Just x  <- lookup v env = eval env x
eval _   term                                   = term

fv env vs (Var s) | s `elem` vs            = []
                  | Just x <- lookup s env = fv env (s:vs) x
                  | otherwise              = [s]
fv env vs (Lam (s, _) f) = fv env (s:vs) f
fv env vs (App x y)      = fv env vs x `union` fv env vs y
fv env vs (Ifz x y z)    = fv env vs x `union` fv env vs y `union` fv env vs z

newName x ys = head $ filter (`notElem` ys) $ (s ++) . show <$> [1..] where
  s = dropWhileEnd isDigit x

rename x x1 term = case term of
  Var s | s == x    -> Var x1
        | otherwise -> term
  Lam (s, t) b
        | s == x    -> term
        | otherwise -> Lam (s, t) (rec b)
  App a b           -> App (rec a) (rec b)
  where rec = rename x x1
\end{code}

== User Interface ==

This is slightly different from our previous demo because our typing algorithm
returns a hopefully helpful message instead of `Nothing` on error.

\begin{code}
#ifdef __HASTE__
main = withElems ["input", "output", "evalB", "resetB", "resetP"] $
  \[iEl, oEl, evalB, resetB, resetP] -> do
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
        Right t -> (out ++ show (eval lets term) ++ "\n", env)
      TopLet s term -> case typeOf gamma term of
        Left m  -> (concat
           [out, "type error: ", show term, ": ", m, "\n"], env)
        Right t  -> (out ++ "[" ++ s ++ ":" ++ show t ++ "]\n",
          ((s, generalize [] t):gamma, (s, term):lets))
  reset
  resetB `onEvent` Click $ const reset
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
              print $ eval lets term
          redo
        Right (TopLet s term) -> case typeOf gamma term of
          Left msg -> putStrLn ("bad type: " ++ msg) >> redo
          Right t  -> do
            putStrLn $ "[" ++ s ++ " : " ++ show t ++ "]"
            repl ((s, generalize [] t):gamma, (s, term):lets)

main = repl ([], [])
#endif
\end{code}
