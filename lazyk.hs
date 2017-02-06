import Data.Char
import System.Console.Readline
import Text.ParserCombinators.Parsec

import Debug.Trace

infixl 5 :@
data Term = Leaf String | Term :@ Term

instance Show Term where
  show (Leaf s)  = s
  show (l :@ r)  = show l ++ showR r where
    showR t@(_ :@ _) = "(" ++ show t ++ ")"
    showR t          = show t

instance Eq Term where
  Leaf s == Leaf t = s == t
  a :@ b == c :@ d = a == c && b == d
  _      == _      = False

ccexpr :: Parser Term
ccexpr = do
  xs <- many expr
  pure $ case xs of
    [] -> Leaf "I"
    _  -> foldl1 (:@) xs

expr     = const (Leaf "I") <$> char 'i' <|> expr'

iotaexpr = const (Leaf "i") <$> char 'i' <|> expr'

expr' = const (Leaf "S") <$> char 's'
    <|> const (Leaf "K") <$> char 'k'
    <|> jotRev . reverse <$> many1 (oneOf "01")
    <|> Leaf . pure <$> letter
    <|> between (char '(') (char ')') ccexpr
    <|> (char '`' >> (:@) <$> expr <*> expr)
    <|> (char '*' >> (:@) <$> iotaexpr <*> iotaexpr)

jotRev []       = Leaf "I"
jotRev ('0':js) = jotRev js :@ Leaf "S" :@ Leaf "K"
jotRev ('1':js) = Leaf "S" :@ (Leaf "K" :@ jotRev js)

data Top = Super String [String] Term | Run Term

top :: Parser Top
top = do
  t <- try super <|> Run <$> ccexpr
  eof
  pure t

super = do
  name <- pure <$> letter
  args <- (pure <$>) <$> many letter
  char '='
  Super name args <$> ccexpr

main = repl []

eval env t = f t [] where
  f (m :@ n) stack = f m (n:stack)
  f (Leaf s) stack | Just t <- lookup s env = f t stack
  f (Leaf "I") (n:stack)     = f n stack
  f (Leaf "K") [x]           = Leaf "K" :@ x
  f (Leaf "K") (x:_:stack)   = f x stack
  f (Leaf "S") [x]           = Leaf "S" :@ x
  f (Leaf "S") [x, y]        = Leaf "S" :@ x :@ y
  f (Leaf "S") (x:y:z:stack) = f (x :@ z :@ (y :@ z)) stack
  f (Leaf "i") (x:stack)     = f (x :@ Leaf "S" :@ Leaf "K") stack
  f t@(Leaf _) stack         = foldl (:@) t stack

norm env term = case eval env term of
  Leaf t -> Leaf t
  m :@ n -> rec m :@ rec n
  where rec = norm env

simpleBracketAbs args = f (reverse args) where
  f [] t = t
  f (x:xs) (Leaf n) | x == n    = f xs $ Leaf "I" 
                    | otherwise = f xs $ Leaf "K" :@ Leaf n
  f (x:xs) (m :@ n)             = f xs $ Leaf "S" :@ f [x] m :@ f [x] n

bracketAbs args = f (reverse args) where
  f [] t = t
  f (x:xs) t = f xs $ case t of
    Leaf "S" :@ Leaf "K" :@ _          -> Leaf "S" :@ Leaf "K"
    m | m `lacks` x                    -> Leaf "K" :@ m
    Leaf n | x == n                    -> Leaf "I" 
    m :@ Leaf n | n == x, m `lacks` x  -> m
    Leaf n0 :@ m :@ Leaf n1 | n0 == x, n1 == x ->
      f [x] $ Leaf "S" :@ Leaf "S" :@ Leaf "K" :@ Leaf x :@ m
    m :@ (n :@ l) | isComb m, isComb n -> f [x] $ Leaf "S" :@ f [x] m :@ n :@ l
    m :@ n :@ l   | isComb m, isComb l -> f [x] $ Leaf "S" :@ m :@ f [x] l :@ n
    m :@ l0 :@ (n :@ l1) | l0 == l1, isComb m, isComb n ->
      f [x] $ Leaf "S" :@ m :@ n :@ l0
    m :@ n                             -> Leaf "S" :@ f [x] m :@ f [x] n

  isComb (Leaf m) = m `notElem` args
  isComb (m :@ n) = isComb m && isComb n

lacks (Leaf m) s = m /= s
lacks (m :@ n) s = lacks m s && lacks n s

repl env = do
  ms <- readline "> "
  case ms of
    Nothing -> putStrLn ""
    Just s  -> do
      addHistory s
      case parse top "" s of
        Left err  -> do
          putStrLn $ "parse error: " ++ show err
          repl env
        Right sup@(Super s args rhs) -> do
          let t = bracketAbs args rhs
          putStrLn $ s ++ "=" ++ show t
          repl ((s, t):env)
        Right (Run term) -> do
          print $ norm env term
          repl env
