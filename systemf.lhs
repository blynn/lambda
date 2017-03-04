= System F =

link:simply.html[Simply typed lambda calculus] is restrictive. The
link:hm.html[let-polymorphism of Hindley-Milner] gives us more breathing room,
but can we do better?

System F frees the type system further by introducing parts of lambda calculus
at the type level. We have 'type abstraction' terms and 'type application'
terms, which define and apply functions that take types as arguments and return
terms. At the same time, System F remains normalizing.

System F is rich enough that the self-application `\x.x x` is typable.

[pass]
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
<script src="systemf.js"></script>
<p><button id="evalB">Run</button>
<button id="resetB">Reset</button>
<button id="pairB">Pair</button>
</p>
<p><textarea style="border: solid 2px; border-color: #999999" id="input" rows="12" cols="80">
</textarea></p>
<p><textarea id="output" rows="12" cols="80" readonly></textarea></p>
<textarea id="resetP" hidden>
id=\X x:X.x -- Polymorphic identity.
xx=\x:forall X.X->X.x[forall X.X->X] x  -- Self-application.
xx id
iter2   = \X f:X->X x:X.f(f x)
iter4   = \X. iter2 [X->X] (iter2 [X])
iter256 = \X. iter4 [X->X] (iter4 [X])  -- 4^4 = 256.
0    = \X s:X->X z:X.z  -- Church numerals.
succ = \n:(forall X.(X->X)->X->X) X s:X->X z:X.s(n[X] s z)
iter4 [forall X.(X->X)->X->X] succ 0
iter256 [forall X.(X->X)->X->X] succ 0
</textarea>
<textarea id="pairP" hidden>
0    = \X s:X->X z:X.z
succ = \n:(forall X.(X->X)->X->X) X s:X->X z:X.s(n[X] s z)
pair  = \X Y x:X y:Y Z f:X->Y->Z.f x y
fst = \X Y p:forall Z.(X->Y->Z)->Z.p [X] (\x:X y:Y.x)
snd = \X Y p:forall Z.(X->Y->Z)->Z.p [Y] (\x:X y:Y.y)
p02 = pair [forall X.(X->X)->X->X] [forall X.(X->X)->X->X] 0 (succ (succ 0))
fst [forall X.(X->X)->X->X] [forall X.(X->X)->X->X] p02
snd [forall X.(X->X)->X->X] [forall X.(X->X)->X->X] p02
</textarea>
<textarea id="listP" hidden>
nil  = \X.(\R c:X->R->R n:R.n)
cons = \X h:X t:forall R.(X->R->R)->R->R.(\R c:X->R->R n:R.c h (t [R] c n))
</textarea>
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

If we focus on the types, System F is a gentle generalization of
link:hm.html[the Hindley-Milner system]. In the latter, any universal
quantifiers `(∀)` must appear at the beginning of type, meaning they are scoped
over the entire type. In System F, they can be arbitrary scoped. For example.
`(∀X.X->X)->(∀X.X->X)` is a System F type, but not a Hindley-Milner type, while
`∀X.(X->X)->(X->X)` is a type in both systems.

This seemingly small change has deep consequences.
link:hm.html[Recall Hindley-Milner] allows:

------------------------------------------------------------------------------
let id = \x.x in id succ(id 0)
------------------------------------------------------------------------------

but rejects:

------------------------------------------------------------------------------
(\f.f succ(f 0)) (\x.x)
------------------------------------------------------------------------------

This is because algorithm W assigns `x` a type variable, say `X`, then finds
the conflicting constraints `X = Nat` and `X = Nat -> Nat`. A locally scoped
generalized variable fixes this by causing fresh type variables to be generated
for each use, so the resulting constraints, say, `X = Nat` and
`Y = Nat -> Nat`, live and let live. Let-free let-polymorphism!

It's easy to demonstrate this in Haskell. The following fails:

------------------------------------------------------------------------------
(\f->f succ(f 0))(\x->x)
------------------------------------------------------------------------------

We can make it run by enabling an extension, and annotating the identity
function appropriately:

------------------------------------------------------------------------------
:set -XRankNTypes
((\f->f succ(f 0)) :: ((forall x.x->x)->Int))(\x->x)
------------------------------------------------------------------------------

We write type abstractions as lambda abstractions without a type annotation.
The simplest example is the 'polymorphic identity function':

------------------------------------------------------------------------------
id=\X.\x:X.x
------------------------------------------------------------------------------

For clarity, we capitalize the first letter of type variables in our examples.

The above represents a function that takes a type, and then returns
an identity function for that type.

We write type applications like term applications except we surround the
argument with square brackets. For example:

------------------------------------------------------------------------------
id [Nat] 42
------------------------------------------------------------------------------

evaluates to 42.

== Type spam ==

Our new features have a heavy price tag. In System F, type reconstruction
is undecidable. We must add explicit types to every binding in every lambda
abstraction. Moreover, applying type abstractions require us to state even more
types.

Haskell magically fills in missing types if enough hints are given, which is
why our example above got away with relatively little annotation. Our
implementation of System F lacks this ability, so we must always supply types.
(This is why we used Haskell above instead of our System F interpreter!)

Because types must frequently be specified, few practical languages are built
on System F. (Perhaps it's also because System F is a relatively recent
discovery?) The Hindley-Milner system is often preferable, due to type
inference.

However, behind the scenes, modern Haskell is an extension of System F.
Certain language features require type annotation, and they generate unseen
intermediate code full of type abstractions and type applications.

In practice, type operators make types less eye-watering:

------------------------------------------------------------------------------
Bool = forall X.X->X->X
Nat = forall X.(X->X)->X->X
Pair X Y = forall Z.(X->Y->Z)->Z
List X = forall R.(X->R->R)->R->R 
------------------------------------------------------------------------------

== Definitions ==

We replace the `GV` constructor representing Hindley-Milner generalized
variables with its scoped version `Forall`.

We add type applications and type abstractions to the `Term` data type. A type
application is like a term application, except it expects a type as input (and
during printing should enclose it within square brackets). A type abstraction
is like a term abstraction, except its variable, being a type variable, has no
type annotation.

\begin{code}
{-# LANGUAGE CPP #-}
#ifdef __HASTE__
import Haste.DOM
import Haste.Events
#else
import System.Console.Readline
#endif
import Control.Monad
import Data.Char
import Data.List
import Text.ParserCombinators.Parsec

data Type = TV String | Forall String Type | Type :-> Type deriving Eq
data Term = Var String | App Term Term | Lam (String, Type) Term
  | Let String Term Term
  | TLam String Term | TApp Term Type

instance Show Type where
  show (TV s) = s
  show (Forall s t) = '\8704':(s ++ "." ++ show t)
  show (t :-> u) = showL t ++ " -> " ++ showR u where
    showL (Forall _ _) = "(" ++ show t ++ ")"
    showL (_ :-> _)    = "(" ++ show t ++ ")"
    showL _            = show t
    showR (Forall _ _) = "(" ++ show u ++ ")"
    showR _            = show u

instance Show Term where
  show (Lam (x, t) y)    = "\0955" ++ x ++ showT t ++ showB y where
    showB (Lam (x, t) y) = " " ++ x ++ showT t ++ showB y
    showB expr           = '.':show expr
    showT (TV "_")       = ""
    showT t              = ':':show t
  show (TLam s t)        = "\0955" ++ s ++ showB t where
    showB (TLam s t)     = " " ++ s ++ showB t
    showB expr           = '.':show expr
  show (Var s)     = s
  show (App x y)   = showL x ++ showR y where
    showL (Lam _ _) = "(" ++ show x ++ ")"
    showL _         = show x
    showR (Var s)   = ' ':s
    showR _         = "(" ++ show y ++ ")"
  show (TApp x y)  = showL x ++ "[" ++ show y ++ "]" where
    showL (Lam _ _) = "(" ++ show x ++ ")"
    showL _         = show x
  show (Let x y z) =
    "let " ++ x ++ " = " ++ show y ++ " in " ++ show z
\end{code}

== Parsing ==

Parsing is trickier because elements of lambda calculus have invaded the
type level. For each abstraction, we look for a type binding to determine
if it's at the term or type level. For applications, we look for square
brackets to decide.

We follow Haskell and accept `forall` as well as the harder-to-type `∀` symbol.
Conventionally, the arrow type operator `->` has higher precedence.

We accept inputs that omit all but the last period and all but the first
quantifier in a sequence of universal quantified type variables, an
abbreviation similar to the one we've been using in sequences of abstractions.

\begin{code}
data FLine = Empty | TopLet String Term | Run Term deriving Show

line :: Parser FLine
line = (((eof >>) . pure) =<<) . (ws >>) $ option Empty $ do
  t <- term
  option (Run t) $ str "=" >> TopLet (getV t) <$> term where
  getV (Var s) = s
  term = letx <|> lam <|> app
  letx = Let <$> (str "let" >> v) <*> (str "=" >> term)
    <*> (str "in" >> term)
  lam = flip (foldr pickLam) <$> between lam0 lam1 (many1 vt) <*> term where
    lam0 = str "\\" <|> str "\0955"
    lam1 = str "."
    vt   = (,) <$> v <*> optionMaybe (str ":" >> typ)
    pickLam (s, Nothing) = TLam s
    pickLam (s, Just t)  = Lam (s, t)
  typ = forallt <|> fun
  forallt = flip (foldr Forall) <$> between fa0 fa1 (many1 v) <*> fun where
    fa0 = str "forall" <|> str "\8704"
    fa1 = str "."
  fun = ((TV <$> v)
    <|> between (str "(") (str ")") typ) `chainr1` (str "->" >> pure (:->))
  app = termArg >>= moreArg
  termArg = (Var <$> v) <|> between (str "(") (str ")") term
  moreArg t = option t $ ((App t <$> termArg)
    <|> (TApp t <$> between (str "[") (str "]") typ)) >>= moreArg
  v = try $ do
    s <- many1 alphaNum
    when (s `elem` words "let in forall") $ fail "unexpected keyword"
    ws
    pure s
  str = try . (>> ws) . string
  ws = spaces >> optional (try $ string "--" >> many anyChar)
\end{code}

== Typing ==

We've abandoned type inference, which simplifies typing. Nonetheless, we
must handle the new terms. Type abstractions are trivial. As for type
applications, once again, a routine used at the term level must now be written
at the type level: we must rename type variables when they conflict with free
type variables.

\begin{code}
typeOf gamma t = case t of
  Var s -> lookup s gamma
  App x y -> do
    tx <- rec x
    ty <- rec y
    case tx of
      ty' :-> tz | ty == ty' -> pure tz
      _                      -> Nothing
  Lam (x, t) y -> do
    u <- typeOf ((x, t):gamma) y
    pure $ t :-> u
  TLam s t -> Forall s <$> typeOf gamma t
  TApp x y -> do
    tx <- typeOf gamma x
    case tx of
      Forall s t -> pure $ beta t where
        beta (TV v) | s == v         = y
                    | otherwise      = TV v
        beta (Forall u v)
                    | s == u         = Forall u v
                    | u `elem` fvs   = let u1 = newName u fvs in
                      Forall u1 $ beta $ tRename u u1 v
                    | otherwise      = Forall u $ beta v
        beta (m :-> n)               = beta m :-> beta n
        fvs = tfv [] y
      _          -> Nothing
  Let s t u -> do
    tt <- typeOf gamma t
    typeOf ((s, tt):gamma) u
  where rec = typeOf gamma

tfv vs (TV s) | s `elem` vs = []
              | otherwise   = [s]
tfv vs (x :-> y)            = tfv vs x `union` tfv vs y
tfv vs (Forall s t)         = tfv (s:vs) t

tRename x x1 ty = case ty of
  TV s | s == x    -> TV x1
       | otherwise -> ty
  Forall s t
       | s == x    -> ty
       | otherwise -> Forall s (rec t)
  a :-> b          -> rec a :-> rec b
  where rec = tRename x x1
\end{code}

== Evaluation ==

Evaluation remains about the same. We erase types as we go.

As usual, the function `eval` takes us to weak head normal form:

\begin{code}
eval env (Let x y z) = eval env $ beta (x, y) z
eval env (App m a) = let m' = eval env m in case m' of
  Lam (v, _) f -> eval env $ beta (v, a) f
  _ -> App m' $ eval env a
eval env (TApp m _) = eval env m
eval env (TLam _ t) = eval env t
eval env term@(Var v) | Just x  <- lookup v env = eval env x
eval _   term                                   = term

beta (v, a) f = case f of
  Var s | s == v       -> a
        | otherwise    -> Var s
  Lam (s, _) m
        | s == v       -> Lam (s, TV "_") m
        | s `elem` fvs -> let s1 = newName s fvs in
          Lam (s1, TV "_") $ rec $ rename s s1 m
        | otherwise    -> Lam (s, TV "_") (rec m)
  App m n              -> App (rec m) (rec n)
  TLam s t             -> TLam s (rec t)
  TApp t ty            -> TApp (rec t) ty
  Let x y z            -> Let x (rec y) (rec z)
  where
    fvs = fv [] a
    rec = beta (v, a)

fv vs (Var s) | s `elem` vs = []
              | otherwise   = [s]
fv vs (Lam (s, _) f)        = fv (s:vs) f
fv vs (App x y)             = fv vs x `union` fv vs y
fv vs (Let _ x y)           = fv vs x `union` fv vs y
fv vs (TLam _ t)            = fv vs t
fv vs (TApp x _)            = fv vs x

newName x ys = head $ filter (`notElem` ys) $ (s ++) . show <$> [1..] where
  s = dropWhileEnd isDigit x

rename x x1 term = case term of
  Var s | s == x    -> Var x1
        | otherwise -> term
  Lam (s, t) b
        | s == x    -> term
        | otherwise -> Lam (s, t) (rec b)
  App a b           -> App (rec a) (rec b)
  Let a b c         -> Let a (rec b) (rec c)
  TLam s t          -> TLam s (rec t)
  TApp a b          -> TApp (rec a) b
  where rec = rename x x1
\end{code}

The function `norm` recurses to find the normal form:

\begin{code}
norm env term = case eval env term of
  Var v        -> Var v
  Lam (v, _) m -> Lam (v, TV "_") (rec m)
  App m n      -> App (rec m) (rec n)
  Let x y z    -> Let x (rec y) (rec z)
  TApp m _     -> rec m
  TLam _ t     -> rec t
  where rec = norm env
\end{code}

== User Interface ==

The less said the better.

\begin{code}
#ifdef __HASTE__
main = withElems ["input", "output", "evalB", "resetB", "resetP",
    "pairB", "pairP"] $
  \[iEl, oEl, evalB, resetB, resetP, pairB, pairP] -> do
  let
    reset = getProp resetP "value" >>= setProp iEl "value" >> setProp oEl "value" ""
    run (out, env) (Left err) =
      (out ++ "parse error: " ++ show err ++ "\n", env)
    run (out, env@(gamma, lets)) (Right m) = case m of
      Empty      -> (out, env)
      Run term   -> case typeOf gamma term of
        Nothing -> (out ++ "type error: " ++ show term ++ "\n", env)
        Just t  -> (out ++ show (norm lets term) ++ "\n", env)
      TopLet s term -> case typeOf gamma term of
        Nothing -> (out ++ "type error: " ++ show term ++ "\n", env)
        Just t  -> (out ++ "[" ++ s ++ ":" ++ show t ++ "]\n",
          ((s, t):gamma, (s, term):lets))
  reset
  resetB `onEvent` Click $ const reset
  pairB `onEvent` Click $ const $
    getProp pairP "value" >>= setProp iEl "value" >> setProp oEl "value" ""
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
            Nothing -> putStrLn "type error"
            Just t -> do
              putStrLn $ "[type = " ++ show t ++ "]"
              print $ norm lets term
          redo
        Right (TopLet s term) -> case typeOf gamma term of
          Nothing -> putStrLn "type error" >> redo
          Just t -> do
            putStrLn $ "[type = " ++ show t ++ "]"
            repl ((s, t):gamma, (s, term):lets)

main = repl ([], [])
#endif
\end{code}

== Booleans, Integers, Pairs, Lists ==

Hindley-Milner supports Church-encodings of booleans, integers, pairs, and
lists. System F is more general, so of course supports them too. 
However, we must be explicit about types. With Hindley-Milner, globally scoped
universal quantifiers for all type variables are implied. With System F,
our terms must start with type abstractions or a term abstraction annotated
with a universal type:

------------------------------------------------------------------------------
true  = \X x:X y:X.x
false = \X x:X y:X.y
not   = \b:forall X.X->X->X X t:X f:X.b [X] f t
0    = \X s:X->X z:X.z
succ = \n:(forall X.(X->X)->X->X) X s:X->X z:X.s(n[X] s z)
pair = \X Y x:X y:Y Z f:X->Y->Z.f x y
fst  = \X Y p:forall Z.(X->Y->Z)->Z.p [X] (\x:X y:Y.x)
snd  = \X Y p:forall Z.(X->Y->Z)->Z.p [Y] (\x:X y:Y.y)
nil  = \X.(\R.\c:X->R->R.\n:R.n)
cons = \X h:X t:forall R.(X->R->R)->R->R.(\R c:X->R->R n:R.c h (t [R] c n))
------------------------------------------------------------------------------

== Apply yourself! ==

In our previous systems, the term `\x.x x` has been untypable. No longer!
Self-application can be expressed in System F:

------------------------------------------------------------------------------
\x:forall X.X->X.x[forall X.X->X] x
------------------------------------------------------------------------------

Of course, self-application of self-application (`(\x.x x)(\x.x x)`) remains
untypable, because it has no normal form.

== Information hiding ==

It turns out universal types can represent 'existential types',
These types can enforce information hiding. For example, we can give
the programmer access to an API but forbid access to the implementation
details.

Knowledge is power. Languages with simpler type systems still benefit if their
designers know System F. For example, Haskell 98 is strictly Hindley-Milner,
but
http://homepages.dcc.ufmg.br/~camarao/fp/articles/lazy-state.pdf[researchers
exploited existential types to design and prove theorems about a language
extension] enabling
https://hackage.haskell.org/package/base/docs/Control-Monad-ST.html[the ST
monad].
