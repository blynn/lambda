= Halfway to Haskell =

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
three = succ (succ (succ 0))
add = fix (\f m n.ifz m then n else f (pred m) (succ n))
mul = fix (\f t m n.ifz m then t else f (add t n) (pred m) n) 0
id = \x.x
add two three
mul three three
id succ (id three)
</textarea>
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Some presentations of PCF also add the base type `Bool` along with constants
`True`, `False` and replace `ifz` with `if` and `iszero`. We already
demonstrated link:simply.html[a simply typed lambda calculus with 2 base
types], so we work with the `Nat`-only version here.

If this were all we were doing, our code here would be about the same as our
previous example. Instead, we take this opportunity to introduce 'type
inference', also known as 'type reconstruction'. We implement an algorithm that
returns the most general type of a given expression, even if all type
information is omitted. The algorithm fails if and only the given expression
cannot be well-typed.

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

data Type = Nat | TV String | Type :-> Type | GV String deriving Eq
data Term = Var String | App Term Term | Lam (String, Type) Term
  | Ifz Term Term Term

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

data PCFLine = Empty | Let String Term | Run Term

line :: Parser PCFLine
line = (((eof >>) . pure) =<<) . (ws >>) $ option Empty $ do
  t <- term
  option (Run t) $ str "=" >> Let (getV t) <$> term where
  getV (Var s) = s
  term = ifz <|> lam <|> app
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
    vt   = do
      s <- v
      t <- option (TV "_") $ str ":" >> typ
      pure $ (s, t)
  typ = ((str "Nat" >> pure Nat) <|> (TV <$> v)
    <|> between (str "(") (str ")") typ)
      `chainr1` (str "->" >> pure (:->))
  app = foldl1' App <$> many1 ((Var <$> v) <|> between (str "(") (str ")") term)
  v   = try $ do
    s <- many1 alphaNum
    when (s `elem` words "ifz then else") $ fail "unexpected keyword"
    ws
    pure s
  str = try . (>> ws) . string
  ws = spaces >> optional (try $ string "--" >> many anyChar)
\end{code}

== Type Inference ==

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

instantiate ty i = f ty [] i where
  f ty m i = case ty of
    GV s | Just t <- lookup s m -> (t, m, i)
         | otherwise            -> (x, (s, x):m, i + 1) where
           x = TV ('_':show i)
    t :-> u -> (t' :-> u', m'', i'') where
      (t', m' , i')  = f t m  i
      (u', m'', i'') = f u m' i'
    _       -> (ty, m, i)

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

freeTV (a :-> b) = freeTV a ++ freeTV b
freeTV (TV tv)   = [tv]
freeTV _         = []

subst (x, t) (TV y) | y == x = t
subst (x, t) (a :-> b) = (subst (x, t) a :-> subst (x, t) b)
subst (x, t) ty = ty

typeOf gamma term = foldl' ren ty <$> unify cs where
  (ty, cs, _) = gather gamma 0 term
  ren ty (s, t) = case ty of
    a :-> b         -> ren a (s, t) :-> ren b (s, t)
    TV tv | tv == s -> t
    _               -> ty

generalize ty = case ty of
  TV s    -> GV s
  s :-> t -> generalize s :-> generalize t
  _       -> ty
\end{code}

== Evaluation ==

\begin{code}
eval env (Ifz x y z) = eval env $ case eval env x of
  Var "0"  -> y
  _        -> z

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

\begin{code}
#ifdef __HASTE__
main = withElems ["input", "output", "evalB", "resetB", "resetP"] $
  \[iEl, oEl, evalB, resetB, resetP] -> do
  let
    reset = getProp resetP "value" >>= setProp iEl "value" >> setProp oEl "value" ""
    run (out, env) (Left err) =
      (out ++ "parse error: " ++ show err ++ "\n", env)
    run (out, env@(gamma, lets)) (Right m) = case m of
      Empty      -> (out, env)
      Run term   -> case typeOf gamma term of
        Left m  -> (concat
           [out, "type error: ", show term, ": ", m, "\n"], env)
        Right t -> (out ++ show (eval lets term) ++ "\n", env)
      Let s term -> case typeOf gamma term of
        Left m  -> (concat
           [out, "type error: ", show term, ": ", m, "\n"], env)
        Right t  -> (out ++ "[" ++ s ++ ":" ++ show t ++ "]\n",
          ((s, generalize t):gamma, (s, term):lets))
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
        Right (Let s term) -> case typeOf gamma term of
          Left msg -> putStrLn ("bad type: " ++ msg) >> redo
          Right t  -> do
            putStrLn $ "[" ++ s ++ " : " ++ show t ++ "]"
            repl ((s, generalize t):gamma, (s, term):lets)

main = repl ([], [])
#endif
\end{code}
