import Text.ParserCombinators.Parsec

data Expr = Atom String | List [Expr]

instance Show Expr where
  show (Atom s) = s
  show (List as) = "(" ++ (unwords $ show <$> as) ++ ")"

f "quote" [x] = x

f "atom" [Atom _ ] = Atom "t"
f "atom" [List []] = Atom "t"
f "atom" [_      ] = List []

f "eq" [List [], List []] = Atom "t"
f "eq" [Atom x , Atom y ] | x == y    = Atom "t"
                          | otherwise = List []
f "eq" [_      , _      ] = List []

f "car" [List (h:_)] = h

f "cdr" [List (_:t)] = List t

f "cons" [x, List y] = List (x:y)

f "cond" []                = List []
f "cond" (List (p:e):rest) = case eval p of
  Atom "t" -> eval $ List e
  _        -> f "cond" rest

isFun "quote" = False
isFun "cond"  = False
isFun _       = True

eval (List (List h:t)) = eval $ List $ eval (List h):t
eval (List (Atom h:t)) | isFun h   = f h $ eval <$> t
                       | otherwise = f h t

expr :: Parser Expr
expr = between spaces spaces $ atom <|> list <|> quote where
  atom = Atom <$> many1 alphaNum
  list = List <$> between (char '(') (char ')') (many expr)
  quote = do
    char '\''
    x <- expr
    return $ List [Atom "quote", x]
