import Data.List
import System.Console.Readline
import Text.ParserCombinators.Parsec.Error

import Control.Monad
import Text.ParserCombinators.Parsec

data Expr = Atom String | List [Expr] | Label String Expr | Bad

instance Show Expr where
  show (Label s _) = s
  show (Atom s)    = s
  show (List as)   = "(" ++ unwords (show <$> as) ++ ")"
  show Bad         = "?"

eval env = g where
  f "quote" [x] = x

  f "atom" [Atom _ ] = Atom "t"
  f "atom" [List []] = Atom "t"
  f "atom" [_      ] = List []

  f "eq" [List [], List []] = Atom "t"
  f "eq" [Atom x , Atom y ] | x == y    = Atom "t"
                            | otherwise = List []
  f "eq" [_      , _      ] = List []

  f "car"  [List (h:_)] = h
  f "cdr"  [List (_:t)] = List t
  f "cons" [h,  List t] = List (h:t)

  f "cond" []                = List []
  f "cond" (List [p, e]:rest) = case g p of Atom "t" -> g e
                                            _        -> f "cond" rest

  f "label" [Atom id, e] = Label id e

  f "defun" [id, ps, e] =
    g $ List [Atom "label", id, List [Atom "lambda", ps, e]]

  f "list" t = List t

  f _ _ = Bad

  -- Convenient, but we can live without these.
  -- g t@(Atom "t") = t
  -- g empty@(List []) = empty

  g (List (Label id e:rest)) = eval ((id, e):env) $ List $ e:rest
  g (Atom s) | Just b <- lookup s env = b
             | otherwise              = Bad
  g (List (List [Atom "lambda", List args, body]:t))
    = eval (zip (fromAtom <$> args) (g <$> t) ++ env) body where
     fromAtom (Atom p) = p
  g (List (List h:t)) = g $ List $ g (List h):t
  g (List (Atom h:t))
    | Just b <- lookup h env                  = g $ List $ b:t
    | elem h $ words "cond quote defun label" = f h t
    | otherwise                               = f h $ g <$> t
  g _ = Bad

expr :: Parser Expr
expr = between spaces spaces $ atom <|> list <|> quote where
  atom = Atom <$> many1 (alphaNum <|> char '.')
  list = List <$> between (char '(') (char ')') (many expr)
  quote = char '\'' >> expr >>= pure . List . (Atom "quote":) . pure
  qquote = do
    char '\''
    x <- expr
    return $ List [Atom "quote", x]

oneExpr = expr >>= (eof >>) . pure

expectParen (Expect "\"(\"") = True
expectParen (Expect "\")\"") = True
expectParen _                = False

addEnv (Label s e) = ((s, e):)
addEnv _           = id

-- Preload definitions such as "(defun cadr (x) (cdr (car x)))".
preload = foldl' f [] $ concat $ genCadr <$> [2..4] where
  f env s = let Right expr = parse oneExpr "" s in addEnv (eval env expr) env

genCadr n = [concat ["(defun c", s, "r (x) (c", [h], "r (c", t, "r x)))"] |
  s@(h:t) <- replicateM n "ad"]

repl pre env = do
  ms <- readline $ if null pre then "> " else ""
  case ms of
    Nothing -> putStrLn ""
    Just s  -> addHistory s >> case parse oneExpr "" $ pre ++ s of
      Left err  -> case find expectParen $ errorMessages err of
        Nothing -> do
          putStrLn $ "parse error: " ++ show err
          repl "" env
        _ -> repl (pre ++ s) env
      Right expr -> do
        let r = eval env expr
        print r
        repl "" $ addEnv r env

main = repl "" preload
