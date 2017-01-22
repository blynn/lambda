= Lambda Calculus =

From http://www.cs.yale.edu/homes/hudak/CS201S08/lambda.pdf['A Brief and
Informal Introduction to the Lambda Calculus'] by Paul Hudak, and
and https://en.wikipedia.org/wiki/Church_encoding[Wikipedia's entry on Church
encoding]:

------------------------------------------------------------------------------
true = \x y -> x
false = \x y -> y
if = \p x y -> p x y
0 = \f x -> x
1 = \f x -> f x
succ = \n f x -> f(n f x)
pred = \n f x -> n(\g h -> h (g f)) (\u -> x) (\u ->u)
add = \m n f x -> m f(n f x)
mul = \m n f -> m(n f)
sub = \m n -> (n pred) m
is0 = \n -> n (\x -> false) true
and = \p q -> p q p
le = \m n -> is0 (sub m n)
eq = \m n -> and (le m n) (le n m)
Y = \f -> (\x ->f(x x))(\x -> f(x x))
factrec = \n -> if (is0 n) 1 (mul n (factrec (pred n)))
fact = Y(\f n -> if (is0 n) 1 (mul n (f (pred n))))
------------------------------------------------------------------------------

\begin{code}
import Data.Char
import Data.List
import System.Console.Readline
import Text.ParserCombinators.Parsec

data Expr = Var String | App Expr Expr | Lam String Expr

instance Show Expr where
  show (Lam x y)  = "\0955" ++ x ++ "." ++ show y
  show (Var s)    = s
  show (App x y)  = showL x ++ showR y where
    showL (Lam _ _ ) = "(" ++ show x ++ ")"
    showL _          = show x
    showR (Var s)    = ' ':s
    showR _          = "(" ++ show y ++ ")"

line :: Parser (String, Expr)
line = do
  spaces
  t <- term
  r <- option ("", t) $ str "=" >> (,) (getV t) <$> term
  eof
  pure r where
  getV (Var s) = s
  term = lam <|> app
  lam = flip (foldr Lam) <$> between (str "\\") (str "->") (many1 v) <*> term
  app = foldl1' App <$> many1 ((Var <$> v) <|> between (str "(") (str ")") term)
  v   = many1 alphaNum >>= (spaces >>) . pure
  str = (>> spaces) . string

eval env expr@(App x a) | Lam v f <- eval env x   = let
  beta (Var s)   | s == v         = a
                 | otherwise      = Var s
  beta (Lam s y) | s == v         = Lam s y
                 | s `elem` frees = let s1 = newName s frees in
                    Lam s1 $ beta $ rename s s1 y
                 | otherwise      = Lam s (beta y)
  beta (App x y)                  = App (beta x) (beta y)
  frees = fv [] a
  fv vs (Var s) | s `elem` vs            = []
                | Just x <- lookup s env = fv (s:vs) x
                | otherwise              = [s]
  fv vs (App x y)                        = fv vs x `union` fv vs y
  fv vs (Lam s f)                        = fv (s:vs) f
  in eval env $ beta f
eval env expr@(Var v)   | Just x  <- lookup v env = eval env x
eval _   expr                                     = expr

newName x ys = head $ filter (`notElem` ys) $ (s ++) . show <$> [1..] where
  s = dropWhileEnd isDigit x

rename x x1 expr = case expr of
  Var s   | s == x    -> Var x1
          | otherwise -> expr
  Lam s b | s == x    -> expr
          | otherwise -> Lam s (rec b)
  App a b             -> App (rec a) (rec b)
  where rec = rename x x1

norm env expr = case eval env expr of
  App x y -> App x $ norm env y
  Lam v f -> Lam v $ norm env f
  Var x   -> Var x
         
main = repl []

repl env = do
  ms <- readline "> "
  case ms of
    Nothing -> putStrLn ""
    Just s  -> do
      addHistory s
      case parse line "" s of
        Left err  -> do
          putStrLn $ "parse error: " ++ show err
          repl env
        Right ("", expr) -> do
          print $ norm env expr
          repl env
        Right (s,  expr) -> repl ((s, expr):env)
\end{code}
