{-# LANGUAGE TemplateHaskell #-}
import Test.QuickCheck.All
import Control.Monad
import Data.Char
import Data.List
import Data.Maybe
import System.Console.Readline
import System.Environment
import System.IO
import Text.ParserCombinators.Parsec
import Text.Read

infixl 5 :@
data Term = Var String | Term :@ Term | Lam String Term

vireo = t where
  Right t = parse ccexpr "" "s(k(s(k(s(k(s(k(ss(kk)))k))s))(s(skk))))k"

skk = Var "s" :@ Var "k" :@ Var "k"
vsk = vireo :@ Var "s" :@ Var "k"

ccexpr :: Parser Term
ccexpr = do
  xs <- many expr
  pure $ case xs of
    [] -> skk
    _  -> foldl1 (:@) xs

expr     = const skk <$> char 'i' <|> expr'
iotaexpr = const vsk <$> char 'i' <|> expr'
expr' = jotRev . reverse <$> many1 (oneOf "01")
    <|> const skk <$> char 'I'
    <|> Var . pure . toLower <$> oneOf "KS"
    <|> Var . pure <$> letter
    <|> between (char '(') (char ')') ccexpr
    <|> (char '`' >> (:@) <$> expr <*> expr)
    <|> (char '*' >> (:@) <$> iotaexpr <*> iotaexpr)
    <|> flip (foldr Lam) <$> between (char '\\' <|> char '\0955') (char '.')
      (many1 $ (:[]) <$> var) <*> ccexpr

var = lookAhead (noneOf "skiSKI") >> letter

jotRev []       = skk
jotRev ('0':js) = jotRev js :@ Var "s" :@ Var "k"
jotRev ('1':js) = Var "s" :@ (Var "k" :@ jotRev js)

data Top = Super String Term | Main Term

top :: Parser Top
top = do
  t <- try super <|> Main <$> ccexpr
  eof
  pure t

super = do
  name <- pure <$> var
  char '='
  Super name <$> ccexpr

parseLine = parse top "" . filter (not . isSpace)

fv vs (Var s) | s `elem` vs = []
              | otherwise   = [s]
fv vs (x :@ y)              = fv vs x `union` fv vs y
fv vs (Lam s f)             = fv (s:vs) f

babs (Lam x e)
  | Var "s" :@ Var"k" :@ _ <- t = Var "s" :@ Var "k"
  | x `notElem` fv [] t         = Var "k" :@ t
  | Var y <- t, x == y          = Var "s" :@  Var "k" :@ Var "k"
  | m :@ Var y <- t, x == y, x `notElem` fv [] m = m
  | Var y :@ m :@ Var z <- t, x == y, x == z =
    babs $ Lam x $ Var "s" :@ Var "s" :@ Var "k" :@ Var x :@ m
  | m :@ (n :@ l) <- t, isComb m, isComb n =
    babs $ Lam x $ Var "s" :@ Lam x m :@ n :@ l
  | (m :@ n) :@ l <- t, isComb m, isComb l =
    babs $ Lam x $ Var "s" :@ m :@ Lam x l :@ n
  | (m :@ l) :@ (n :@ l') <- t, l `noLamEq` l', isComb m, isComb n =
    babs $ Lam x $ Var "s" :@ m :@ n :@ l
  | m :@ n <- t = Var "s" :@ babs (Lam x m) :@ babs (Lam x n)
  where t = babs e
babs (Var s) = Var s
babs (m :@ n) = babs m :@ babs n

sub env (x :@ y) = sub env x :@ sub env y
sub env (Var s)  | s `elem` ["s", "k"]    = Var s
                 | Just t <- lookup s env = sub env t
                 | otherwise              = error $ "no binding for " ++ s

isComb e = null $ fv [] e \\ ["s", "k"]

noLamEq (Var x) (Var y) = x == y
noLamEq (a :@ b) (c :@ d) = a `noLamEq` c && b `noLamEq` d
noLamEq _ _ = False

data RunValue = I Int | S String

run (m :@ n) stack = run m (n:stack)
run (Var "k") (x:_:stack)   = run x stack
run (Var "s") (x:y:z:stack) = run x $ z:(y :@ z):stack
run (Var "+") [x] | I n <- run x [] = I $ 1 + n
run (Var "0") [] = I 0
run (Var "!") (x:y:stack) | I n <- run x [Var "+", Var "0"] = S $ case n of
  256 -> []
  n   -> chr n : t where S t = run y $ Var "!":stack
run (Var ">") (x:y:stack) = S $ chr n : t where
  S t = run y stack
  I n = run x [Var "+", Var "0"]
run (Var ".") [] = S []
run e s = error $ show e

chuList []     = vireo :@ church 256     :@ chuList []
chuList (x:xs) = vireo :@ church (ord x) :@ chuList xs

church 0 = Var "k" :@ skk
church n = Var "s" :@ (Var "s" :@ (Var "k" :@ Var "s") :@ Var "k")
  :@ church (n - 1)

rfold = t where
  Right t = parse ccexpr "" "s(k(s(k(s(ks)(s(k(s(ks)k)))))(s(skk))))k"

rfList s = foldr (\a b -> rfold :@ a :@ b)
  (Var "s" :@ Var "k") $ church . ord <$> s

lazyK t inp = g (t :@ chuList inp) where
  g ti = case run ti [Var "k", Var "+", Var "0"] of
    I 256 -> []
    I n   -> chr n : g (ti :@ (Var "k" :@ skk))

fussyK t inp = s where S s = run t [chuList inp, Var "!"]

crazyL t inp = s where S s = run t [rfList inp, Var ">", Var "."]

succ0 t _ = show n where I n = run t [Var "+", Var "0"]
nat2nat t inp = show n where
  I n = run t [church $ fromMaybe 0 $ readMaybe inp, Var "+", Var "0"]

dumpSK t _ = show t

dumpIota t _ = f t where
  f (x :@ y)  = '*':f x ++ f y
  f (Var "k") = "*i*i*ii"
  f (Var "s") = "*i*i*i*ii"

dumpJot t _ = f t where
  f (x :@ y)  = '1':f x ++ f y
  f (Var "k") = "11100"
  f (Var "s") = "11111000"

dumpUnlambda t _ = f t where
  f (x :@ y)  = '`':f x ++ f y
  f (Var "k") = "k"
  f (Var "s") = "s"

instance Show Term where
  show (Var s)  = s
  show (l :@ r)  = show l ++ showR r where
    showR t@(_ :@ _) = "(" ++ show t ++ ")"
    showR t          = show t

repl lang inp env = do
  let rec = repl lang inp
  ms <- readline "> "
  case ms of
    Nothing -> putStrLn ""
    Just s  -> do
      addHistory s
      case parseLine s of
        Left err  -> do
          putStrLn $ "parse error: " ++ show err
          rec env
        Right (Super s rhs) -> do
          let t = babs rhs
          putStrLn $ s ++ "=" ++ show t
          rec ((s, t):env)
        Right (Main term) -> do
          putStrLn $ lang (sub env $ babs term) inp
          rec env

-- Examples from https://tromp.github.io/cl/lazy-k.html.

rev="1111100011111111100000111111111000001111111000111100111111000111111100011110011111000111111100011110011100111111100011110011111111100011111111100000111111111000001111111100011111111100000111111111000001111111000111111100011110011111000111001111111110000011111110001111001111110001111111110000011100111001111111000111100111111000111100111111000111111100011111110001111111110000011110011100111100111111100011110011111100011111111100000111001111111000111100111111000111100111111000111100111001111111000111111100011110011111000111111100011110011111100011110011111000111111100011111110001111001111100011111110001111001110011111110001111001111100011111110001111001110011111110001111111110000011111111100000111100111111100011110011111100011111110001111001111100011111110001111001111110001111111110000011111111000111110001111110001111111110000011110011100111111100011110011100111001111001111001111111000111111111000001111001111001111111110000011110011111111000111111111000001111111110000011111111000111111111000001111111110000011111110001111111000111100111110001110011111111100000"

pri="K(SII(S(K(S(S(K(SII(S(S(KS)(S(K(S(KS)))(S(K(S(S(KS)(SS(S(S(KS)K))(KK)))))(S(S(KS)(S(KK)(S(KS)(S(S(KS)(S(KK)(S(KS)(S(S(KS)(S(KK)(SII)))(K(SI(KK)))))))(K(S(K(S(S(KS)(S(K(SI))(S(KK)(S(K(S(S(KS)K)(S(S(KS)K)I)(S(SII)I(S(S(KS)K)I)(S(S(KS)K)))))(SI(K(KI)))))))))(S(KK)K)))))))(K(S(KK)(S(SI(K(S(S(S(S(SSK(SI(K(KI))))(K(S(S(KS)K)I(S(S(KS)K)(S(S(KS)K)I))(S(K(S(SI(K(KI)))))K)(KK))))(KK))(S(S(KS)(S(K(SI))(S(KK)(S(K(S(S(KS)K)))(SI(KK))))))(K(K(KI)))))(S(S(KS)(S(K(SI))(SS(SI)(KK))))(S(KK)(S(K(S(S(KS)K)))(SI(K(KI)))))))))(K(K(KI))))))))))(K(KI)))))(SI(KK)))))(S(K(S(K(S(K(S(SI(K(S(K(S(S(KS)K)I))(S(SII)I(S(S(KS)K)I)))))))K))))(S(S(KS)(S(KK)(SII)))(K(SI(K(KI)))))))(SII(S(K(S(S(KS)(S(K(S(S(SI(KK))(KI))))(SS(S(S(KS)(S(KK)(S(KS)(S(K(SI))K)))))(KK))))))(S(S(KS)(S(K(S(KS)))(S(K(S(KK)))(S(S(KS)(S(KK)(SII)))(K(S(S(KS)K)))))))(K(S(S(KS)(S(K(S(S(SI(KK))(KI))))(S(KK)(S(K(SII(S(K(S(S(KS)(S(K(S(K(S(S(KS)(S(KK)(S(KS)(S(K(SI))K))))(KK)))))(S(S(KS)(S(KK)(S(K(SI(KK)))(SI(KK)))))(K(SI(KK))))))))(S(S(KS)(S(K(S(KS)))(S(K(S(KK)))(S(S(KS)(S(KK)(SII)))(K(SI(K(KI))))))))(K(K(SI(K(KI)))))))))(S(K(SII))(S(K(S(K(SI(K(KI))))))(S(S(KS)(S(KK)(SI(K(S(K(S(SI(K(KI)))))K)))))(K(S(K(S(SI(KK))))(S(KK)(SII)))))))))))(K(SI(K(KI))))))))(S(S(KS)K)I)(SII(S(K(S(K(S(SI(K(KI)))))K))(SII)))))"

kk256="k(k(s(skk)(skk)(s(skk)(skk)(s(s(ks)k)(skk)))))"

mustParse s = t where Right (Main t) = parseLine s
prop_rev s = lazyK (mustParse rev)   t == reverse t where t = take 10 s
prop_id s  = lazyK (mustParse "")    s == s
prop_emp s = lazyK (mustParse kk256) s == ""
prop_pri   = "2 3 5 7 11 13" `isPrefixOf` lazyK  (mustParse pri) ""
prop_pri'  = "2 3 5 7 11 13" `isPrefixOf` fussyK (mustParse pri) ""
return []
runAllTests = $quickCheckAll

main = do
  as <- getArgs
  let
    f lang = g lang $ case tail as of
      []    -> ""
      (a:_) -> a
    g lang inp = hSetBuffering stdout NoBuffering >> repl lang inp []
  if null as then g lazyK "Hello, World!" else case head as of
    "n"     -> f succ0
    "n2n"   -> f nat2nat
    "lazy"  -> f lazyK
    "fussy" -> f fussyK
    "crazy" -> f crazyL
    "sk"    -> f dumpSK
    "iota"  -> f dumpIota
    "jot"   -> f dumpJot
    "unl"   -> f dumpUnlambda
    "test"  -> void runAllTests
    bad     -> putStrLn $ "bad command: " ++ bad
