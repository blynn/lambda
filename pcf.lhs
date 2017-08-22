= Let there be let =

We've shown link:hm.lhs[how to infer the type of an expression]. But how does
it apply to definitions? Should the following be legal?

------------------------------------------------------------------------------
id = \x.x
one = id succ(id 0)
------------------------------------------------------------------------------

We infer `id` has type `X -> X` where `X` is a type variable. Should we be
allowed to substitute `Nat -> Nat` for `X` in one occurence of `id` and `Nat`
for the other?

If we can only pick one type constant for `X`, then it seems we must duplicate
code:

------------------------------------------------------------------------------
id = \x.x
id2 = \x.x
one = id succ(id2 0)
------------------------------------------------------------------------------

On the other hand, before type inference, we could expand `one` to:

------------------------------------------------------------------------------
one = ((\x.x) succ)((\x.x) 0)
------------------------------------------------------------------------------

and only then introduce type variables. Afterwards, we can substitute different
types for different uses of a definition.

The same discussion applies to local let expressions. That is, suppose
we allow `let _ = _ in _` anywhere we expect a term. Then it might be
reasonable to permit the following:

------------------------------------------------------------------------------
one = let f = \x.x in f succ (f 0)
------------------------------------------------------------------------------

Otherwise we might be forced to duplicate code:

------------------------------------------------------------------------------
one = let f = \x.x in let g = \x.x in (\f g.f succ(g 0))
------------------------------------------------------------------------------

Allowing different uses of a definition to have different types is called
'let-polymorphism'. We demonstrate it with an interpreter based on
https://en.wikipedia.org/wiki/Programming_Computable_Functions[PCF (Programming
Computable Functions)], a simply typed lambda calculus with the base type `Nat`
with the constant `0` and extended with:

 - `pred`, `succ`: these functions have type `Nat -> Nat` and return the
   predecessor and successor of their input; evaluating `pred 0` anywhere in
   a term returns the `Err` term which represents this exception.

 - `ifz-then-else`: when given 0, an `ifz` expression evaluates to its `then`
   branch, otherwise it evaluates to its `else` branch.

 - `fix`: the fixpoint operator, allowing recursion (but breaking
   normalization).

For convenience, we parse all natural numbers as constants of type `Nat`.
We also provide an `undefined` keyword that throws an error.

Avoiding the fixpoint operator guarantees normalization, that is, we get
programs that must terminate. Even with this restriction, the language is
surprisingly expressive: we can sort lists without `fix`!

[pass]
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
<script src="pcf.js"></script>
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

http://homepages.inf.ed.ac.uk/gdp/publications/LCF.pdf[The original PCF]
lacks type inference and let-polymorphism, unlike
http://www.cse.chalmers.se/edu/year/2010/course/DAT140_Types/PCF.pdf[later
definitions of PCF].

== Memoized type inference ==

We should mention how to evaluate local let definitions.
Suppose we write:

------------------------------------------------------------------------------
\x y.let z = \a b.a in z x y
------------------------------------------------------------------------------

Evaluating this is trivial:

------------------------------------------------------------------------------
eval env (Let x y z) = eval env $ beta (x, y) z
------------------------------------------------------------------------------

That is, we add a new binding to the environment before evaluating the let
body. An easy exercise is to add this to our previous interpreter: after
trivially modify parsing and type-checking, it should just work.

As for type inference, we could treat `let` as a macro: we could fully expand
all let definitions before type checking if we accept that work may be
repeated. For example:

------------------------------------------------------------------------------
let f = \x.x in f succ (f 0)
------------------------------------------------------------------------------

could be expanded to:

------------------------------------------------------------------------------
(\x.x) succ ((\x.x) 0)
------------------------------------------------------------------------------

We would determine the first `(\x.x)` has type `_0 -> _0` where `_0` is a
generated type variable, before deducing further that `_0` must be `Nat ->
Nat`. Afterwards, we would repeat computations to determine that the second
`(\x.x)` has type `_1 -> _1`, before deducing `_1` must be `Nat`.

However, this approach has drawbacks. Functions can be more complicated than
`\x.x` and let expansions can be deeply nested, leading to prohibitively many
repeated computations.  Also, we may one day wish to support a recursive
variant of `let`, where full expansion is impossible.

Better to https://en.wikipedia.org/wiki/Memoization[memoize: cache the results
of a computation for later reuse]. We introduce 'generalized type variables'
for this purpose. A generalized type variable is a placeholder that generates a
fresh type variable on demand.

In our example above, we first use type inference to determine `id` has type `X
-> X` where `X` is a type variable. Next, we mark `X` as a generalized type
variable. Then each time `id` is used in an expression, we replace `X` with a
newly generated ordinary type variable before proceeding with type inference.

== Universally quantified types ==

Memoization is also useful for understanding the theory. Rather than
vaguely say `id` is a sort of macro, we say that `id = \x.x` has type `∀X.X ->
X`.  The symbol `∀` indicates a given type variable is generalized. Lambda
calculus with generalized type variables from let-polymorphism is known as the
'Hindley-Milner type system', or HM for short. Like simply typed lambda
calculus, HM is strongly normalizing.

We might then wonder if this `∀` notation is redundant. Since let definitions
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
functions, but these have little theoretical significance.

The juicy missing pieces are algebraic data types and type classes.

Later versions of Haskell go beyond Hindley-Milner to a variant of
link:systemf.html[System F], and there are plans to go even further.
As a result, type inference is no longer guaranteed to succeed, and often the
programmer must supply annotations to help the type checker.

We would be close to ML if we had chosen eager evaluation instead of lazy
evaluation.

== Definitions ==

Despite the advanced capabilities of HM, we can almost reuse the data
structures of simply typed lambda calculus.
In a way, we could do with less. HM is rich enough that we can get by with
no base types whatsoever. However, we're implementing PCF so we provide `Nat`.

To our data type representing types, we add type variables and generalized type
variables: our `TV` and `GV` constructors. And to our data type representing
terms, we add a `Let` constructor to represent let expressions.

To keep the code simple, we show generalized type variables in a nonstandard
manner: we simply prepend an at sign to the variable name. It's understood
that `(@x -> y) -> @z` really means `∀@x @z.(@x -> y) -> @z`.

Since we follow Haskell's convention by showing non-generalized type variables
for top-level let expressions, under normal operation we'll never show a
generalized type variable. Roughly speaking, we lazily generalize the type
variables of let statements, that is, we store them as ordinary type variables,
and generalize them on demand during evaluation. A generalized type variable
would only be printed if we, say, added a logging statement for debugging.

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
line = (((eof >>) . pure) =<<) . (ws >>) $ option Empty $
    (try $ TopLet <$> v <*> (str "=" >> term)) <|> (Run <$> term) where
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

We add generalized variables to link:hm.html[our implementation of Algorithm W].
The `instantiate` function generates fresh type variables for any generalized
type variables. This time, we write it without the state monad so we can
compare styles. On balance, I prefer the state monad version.

If a variable name is absent from `gamma`, then the term is invalid.
We abuse the `GV` constructor to represent this error.

We're careful with let expressions: we only generalize those type
variables that are absent from `gamma` before recursively calling `gather`.

We always generate a fresh variable for `undefined` so it can fit anywhere.

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

instantiate = f [] where
  f m ty i = case ty of
    GV s | Just t <- lookup s m -> (t, m, i)
         | otherwise            -> (x, (s, x):m, i + 1) where
           x = TV ('_':show i)
    t :-> u -> (t' :-> u', m'', i'') where
      (t', m' , i')  = f m  t i
      (u', m'', i'') = f m' u i'
    _       -> (ty, m, i)

generalize fvs ty = case ty of
  TV s | s `notElem` fvs -> GV s
  s :-> t                -> generalize fvs s :-> generalize fvs t
  _                      -> ty

freeTV (a :-> b) = freeTV a ++ freeTV b
freeTV (TV tv)   = [tv]
freeTV _         = []

unify ((GV s, GV "?"):_)   = Left s
unify []                   = Right []
unify ((s, t):cs) | s == t = unify cs
unify ((TV x, t):cs)
  | x `elem` freeTV t = Left $ "infinite: " ++ x ++ " = " ++ show t
  | otherwise         = ((x, t):) <$> unify (join (***) (subst (x, t)) <$> cs)
unify ((s, TV y):cs)  = unify ((TV y, s):cs)
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

Once we're certain a closed term is well-typed, we can ignore the types and
evaluate as we would in untyped lambda calculus.

If we only wanted the weak head normal form, then we could take shortcuts: we
could assume the first argument to any `ifz`, `pred`, or `succ` is a natural
number. However, we want the normal form, necessitating extra checks.

If we encounter an `Err` term, we propagate it up the tree to halt computation.

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
  Var "fix" -> eval env (App a (App m' a))
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

On error, our typing algorithm now returns a more detailed mesage instead of
`Nothing`.

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
        Right t -> (out ++ "[" ++ s ++ ":" ++ show t ++ "]\n",
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
head  = foldr const undefined
null  = foldr (const . const False) True
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

f x ([y], t) | x < y = ([], x:y:t)
f x (a  , t)         = (a , x:t)
------------------------------------------------------------------------------

Insertion sort immediately follows:

------------------------------------------------------------------------------
sort :: Ord a => [a] -> [a]
sort = foldr ins []
------------------------------------------------------------------------------

We can
https://en.wikipedia.org/wiki/Church_encoding#Represent_the_list_using_right_fold[represent
lists with right folds]. The list is a function, and it acts just like `foldr`
if we give it a folding function and an initial value:

------------------------------------------------------------------------------
nil = \c n->n
con = \h t c n->c h(t c n)
example = con 3(con 1(con 4 nil))
example (:) []            -- [3, 1, 4]
foldr   (:) [] [3, 1, 4]  -- [3, 1, 4]
------------------------------------------------------------------------------

In simply typed lambda calculus, we must fix the type of the fold result.
For example, a list of integers might be represented as right fold that
returns an integer, and we can compute the sum of a list of integers as
follows:

------------------------------------------------------------------------------
nil=\c:I->I->I n:I.n
cons=\h:I t:(I->I->I)->I->I c:I->I->I n:I.c h(t c n)
sum=\xs:(I->I->I)->I->I.xs(\h:I t:I.add h t)0
sum (cons 1 (cons 125 (cons 27 nil)))
------------------------------------------------------------------------------

This is about as far as we can go. Without let-polymorphism, we're stuck with
a single fold return type, limiting what we can achieve.

Hindley-Milner frees us. Thanks to generalized type variables,
a single fold can return any type we want. We can port our Haskell code to
lambda calculus to obtain a sorting function free of `fix`. We do use `fix` in
our less-than function, but in a practical language this would be a built-in
primitive. Alternatively we can use Church numerals, which has a well-known
fix-free less-than-or-equal-to function.

It almost seems we're cheating to avoid explicit loops by piggybacking off the
representation of the list, but this is merely a consequence of our strategy.
When functions represent data, we can perform complex tasks with miraculously
concise code.

[pass]
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
<textarea id="sortP" hidden>
-- Insertion sort sans fixpoint. Slow.
pair=\x y f.f x y
fst=\p.p(\x y.x)
snd=\p.p(\x y.y)
nil=\c n.n
cons=\h t c n.c h(t c n)
null=\l.l(\h t.0)1
head=\l.l(\h t.h)undefined
lt=fix(\f x y.ifz y then 0 else ifz x then 1 else f (pred x) (pred y))
f=\x p.ifz null (fst p) then ifz lt x (head (fst p)) then pair (fst p) (cons x (snd p)) else pair nil (cons x (cons (head (fst p)) (snd p))) else pair nil (cons x (snd p))
ins=\x l.let q = l f (pair (cons x nil) nil) in (ifz null (fst q) then (cons (head (fst q)) (snd q)) else snd q)
sort=\l.l ins nil
sort (cons 3(cons 1(cons 4(cons 1(cons 5 nil)))))
</textarea>
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

//////////////////////////////////////////////////////////////////////////////
I spent hours thinking my code was buggy because the lambda calculus
translation of the following refused to type-check:

  nil = \c n->n
  con = \h t c n->c h(t c n)
  nul = \l->l(\h t->0)1
  broken = \l->con (nul l) (l (\h t->t) nil)
  let l = con 42 nil in con (nul l) (l (\h t->t) nil) (:) []  -- Works.

Eventually I plugged the above into Haskell, where type inference failed with
the same error! I'm still unsure why it fails, but at least I have more
confidence in my code now.
//////////////////////////////////////////////////////////////////////////////

== One step back, ten steps forward ==

Hindley-Milner is considered a sweet spot in the language design space because
type inference is simple and decidable, yet the type system is powerful.
There is a blemish: type inference takes exponential time for certain
pathological caess. Luckily, they never show up in real life.

Still, experience suggests we should weaken Hindley-Milner in practical
programming languages.
https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tldi10-vytiniotis.pdf[Let
should not be generalised] automatically for local bindings.

Removing implicit local let-polymorphism makes it easier to extend the type
system. For example, we can add type classes and sidestep the monomorphism
restriction controversy. At the same time, local let-polymorphism is rarely
used, and in any case, we can trivially declare a type when needed.  In other
words, we can still support local let-polymorphism; it's just no longer
automatic.

Also, the pathological exponential cases disappear, which I think means type
inference is guaranteed to be efficient, provided
https://cs.stackexchange.com/questions/6617/concise-example-of-exponential-cost-of-ml-type-inference[we use abbreviations for repeated parts of types].

But most exciting of all, this small tactical retreat may have a huge
payoff in the hopefully near future:
https://typesandkinds.wordpress.com/2016/07/24/dependent-types-in-haskell-progress-report/[dependent types in Haskell]!
