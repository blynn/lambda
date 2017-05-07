== Lazy K, Crazy L ==

Y=ssk(s(k(ss(s(ssk))))k)
P=\nfx.n(\gh.h(gf))(\u.x)(\u.u)
M=\mnf.m(nf)
z=\n.n(\x.sk)k
Y(\fn.zn(\fx.fx)(Mn(f(Pn))))
[pass]
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
<script src="crazyl.js"></script>
<textarea id="input" rows="16" cols="80">
1111100011111111100000111111111000001111111000111100111111000111111100011110011111000111111100011110011100111111100011110011111111100011111111100000111111111000001111111100011111111100000111111111000001111111000111111100011110011111000111001111111110000011111110001111001111110001111111110000011100111001111111000111100111111000111100111111000111111100011111110001111111110000011110011100111100111111100011110011111100011111111100000111001111111000111100111111000111100111111000111100111001111111000111111100011110011111000111111100011110011111100011110011111000111111100011111110001111001111100011111110001111001110011111110001111001111100011111110001111001110011111110001111111110000011111111100000111100111111100011110011111100011111110001111001111100011111110001111001111110001111111110000011111111000111110001111110001111111110000011110011100111111100011110011100111001111001111001111111000111111111000001111001111001111111110000011110011111111000111111111000001111111110000011111111000111111111000001111111110000011111110001111111000111100111110001110011111111100000
</textarea>
<br>
<button id="evalB">Compile + Run</button>
<br>
<br>
<textarea id="output" rows="1" cols="8" readonly></textarea>
<br>
<b>intermediate form</b>:
<br>
<textarea id="sk" rows="5" cols="80" readonly></textarea>
<br>
<b>wasm</b>:
<br>
<textarea id="asm" rows="8" cols="80" readonly></textarea>
<script type="text/javascript">
function runWasmInts(a) {
  WebAssembly.instantiate(new Uint8Array(a),
    {i:{f:x => Haste.putChar(x), g:Haste.getChar}}).then(x => x.instance.exports.e());
}
</script>
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

\begin{code}
{-# LANGUAGE CPP #-}
#ifdef __HASTE__
{-# LANGUAGE OverloadedStrings #-}
import Haste
import Control.Concurrent.MVar
import Haste.DOM
import Haste.Events
import Haste.Foreign
import Numeric
#else
{-# LANGUAGE TemplateHaskell #-}
import System.Console.Readline
import System.Environment
import System.IO
import Test.QuickCheck.All
#endif
import Control.Monad
import Data.Char
import Data.List
import Data.Maybe
import Text.ParserCombinators.Parsec
import Text.Read

infixl 5 :@
data Term = Var String | Term :@ Term | Lam String Term

vireo = t where
  Right t = parse ccexpr "" "s(k(s(k(s(k(s(k(ss(kk)))k))s))(s(skk))))k"

rfold = t where
  Right t = parse ccexpr "" "s(k(s(k(s(ks)(s(k(s(ks)k)))))(s(skk))))k"

skk = Var "s" :@ Var "k" :@ Var "k"
vsk = vireo :@ Var "s" :@ Var "k"
scBird = Var "s" :@ (Var "s" :@ (Var "k" :@ Var "s") :@ Var "k")

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

--chuList []     = vireo :@ church 256     :@ chuList []
chuList []     = vireo :@ church 256     :@ Var "k"
chuList (x:xs) = vireo :@ church (ord x) :@ chuList xs

church 0 = Var "k" :@ skk
church n = Var "s" :@ (Var "s" :@ (Var "k" :@ Var "s") :@ Var "k")
  :@ church (n - 1)

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

-- Examples from https://tromp.github.io/cl/lazy-k.html.

rev = concat [
  "1111100011111111100000111111111000001111111000111100111111000111111",
  "1000111100111110001111111000111100111001111111000111100111111111000",
  "1111111110000011111111100000111111110001111111110000011111111100000",
  "1111111000111111100011110011111000111001111111110000011111110001111",
  "0011111100011111111100000111001110011111110001111001111110001111001",
  "1111100011111110001111111000111111111000001111001110011110011111110",
  "0011110011111100011111111100000111001111111000111100111111000111100",
  "1111110001111001110011111110001111111000111100111110001111111000111",
  "1001111110001111001111100011111110001111111000111100111110001111111",
  "0001111001110011111110001111001111100011111110001111001110011111110",
  "0011111111100000111111111000001111001111111000111100111111000111111",
  "1000111100111110001111111000111100111111000111111111000001111111100",
  "0111110001111110001111111110000011110011100111111100011110011100111",
  "0011110011110011111110001111111110000011110011110011111111100000111",
  "1001111111100011111111100000111111111000001111111100011111111100000",
  "1111111110000011111110001111111000111100111110001110011111111100000"]

pri = concat [
  "K",
  "(SII(S(K(S(S(K(SII(S(S(KS)(S(K(S(KS)))(S(K(S(S(KS)(SS(S(S(KS)K))(KK)))))",
  "(S(S(KS)(S(KK)(S(KS)(S(S(KS)(S(KK)(S(KS)(S(S(KS)(S(KK)(SII)))",
  "(K(SI(KK)))))))(K(S(K(S(S(KS)(S(K(SI))(S(KK)(S(K(S(S(KS)K)(S(S(KS)K)I)",
  "(S(SII)I(S(S(KS)K)I)(S(S(KS)K)))))(SI(K(KI)))))))))(S(KK)K)))))))(K(S(KK)",
  "(S(SI(K(S(S(S(S(SSK(SI(K(KI))))(K(S(S(KS)K)I(S(S(KS)K)(S(S(KS)K)I))",
  "(S(K(S(SI(K(KI)))))K)(KK))))(KK))(S(S(KS)(S(K(SI))(S(KK)(S(K(S(S(KS)K)))",
  "(SI(KK))))))(K(K(KI)))))(S(S(KS)(S(K(SI))(SS(SI)(KK))))(S(KK)",
  "(S(K(S(S(KS)K)))(SI(K(KI)))))))))(K(K(KI))))))))))(K(KI)))))(SI(KK)))))",
  "(S(K(S(K(S(K(S(SI(K(S(K(S(S(KS)K)I))(S(SII)I(S(S(KS)K)I)))))))K))))",
  "(S(S(KS)(S(KK)(SII)))(K(SI(K(KI)))))))(SII(S(K(S(S(KS)(S(K(S(S(SI(KK))",
  "(KI))))(SS(S(S(KS)(S(KK)(S(KS)(S(K(SI))K)))))(KK))))))(S(S(KS)",
  "(S(K(S(KS)))(S(K(S(KK)))(S(S(KS)(S(KK)(SII)))(K(S(S(KS)K)))))))(K(S(S(KS)",
  "(S(K(S(S(SI(KK))(KI))))(S(KK)(S(K(SII(S(K(S(S(KS)(S(K(S(K(S(S(KS)(S(KK)",
  "(S(KS)(S(K(SI))K))))(KK)))))(S(S(KS)(S(KK)(S(K(SI(KK)))(SI(KK)))))",
  "(K(SI(KK))))))))(S(S(KS)(S(K(S(KS)))(S(K(S(KK)))(S(S(KS)(S(KK)(SII)))",
  "(K(SI(K(KI))))))))(K(K(SI(K(KI)))))))))(S(K(SII))(S(K(S(K(SI(K(KI))))))",
  "(S(S(KS)(S(KK)(SI(K(S(K(S(SI(K(KI)))))K)))))(K(S(K(S(SI(KK))))",
  "(S(KK)(SII)))))))))))(K(SI(K(KI))))))))(S(S(KS)K)I)",
  "(SII(S(K(S(K(S(SI(K(KI)))))K))(SII)))))"]

kk256 = "k(k(s(skk)(skk)(s(skk)(skk)(s(s(ks)k)(skk)))))"

mustParse s = t where Right (Main t) = parseLine s
prop_rev s = lazyK (mustParse rev)   t == reverse t where t = take 10 s
prop_id s  = lazyK (mustParse "")    s == s
prop_emp s = lazyK (mustParse kk256) s == ""
prop_pri   = "2 3 5 7 11 13" `isPrefixOf` lazyK  (mustParse pri) ""
prop_pri'  = "2 3 5 7 11 13" `isPrefixOf` fussyK (mustParse pri) ""

fac = unlines [
  "Y=ssk(s(k(ss(s(ssk))))k)",
  "P=\\nfx.n(\\gh.h(gf))(\\u.x)(\\u.u)",
  "M=\\mnf.m(nf)",
  "z=\\n.n(\\x.sk)k",
  "Y(\\fn.zn(\\fx.fx)(Mn(f(Pn))))"]

prop_fac   = nat2nat (mustParseProgram fac) "5" == "120"

parseProgram s = case mEnv of
  Left err -> Left $ "parse error: " ++ show err
  Right env -> case lookup "*main*" env of
    Nothing -> Left "missing main"
    Just m  -> Right $ sub env m
  where
    mEnv = map f <$> mapM parseLine (lines s)
    f (Super s rhs) = (s,        babs rhs)
    f (Main term)   = ("*main*", babs term)
mustParseProgram s = t where Right t = parseProgram s

#ifndef __HASTE__
return []
runAllTests = $quickCheckAll
#endif
\end{code}

\begin{code}
#ifndef __HASTE__
repl lang inp env = do
  let rec = repl lang inp
  ms <- readline "> "
  case ms of
    Nothing -> putStrLn ""
    Just s  -> do
      addHistory s
      case parseLine s of
        Left err  -> do
          putStrLn $ "parse: " ++ show err
          rec env
        Right (Super s rhs) -> do
          let t = babs rhs
          putStrLn $ s ++ "=" ++ show t
          rec ((s, t):env)
        Right (Main term) -> do
          putStrLn $ lang (sub env $ babs term) inp
          rec env

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
    --"enc"   -> f enc where enc t inp = show $ compile t
    "enc"   -> f enc where enc t inp = show $ gen
    "test"  -> void runAllTests
    bad     -> putStrLn $ "bad command: " ++ bad
#endif
\end{code}

\begin{code}
encodeTree e = gen ++ toArr (length gen) e
addrSucc = 257 * 8
codeSucc = toArr addrSucc scBird
addrVireo = addrSucc + length codeSucc
codeVireo = toArr addrVireo vireo

genChurch = enCom "s" ++ enCom "k" ++ concat [toU32 addrSucc ++ toU32 (n * 8) | n <- [0..255]]

gen = genChurch ++ codeSucc ++ codeVireo

enCom "0" = neg32 1
enCom "+" = neg32 2
enCom "k" = neg32 3
enCom "s" = neg32 4
enCom "<" = neg32 5
enCom "!" = neg32 6
enCom ">" = neg32 7
enCom "." = neg32 8
toArr n (Var a :@ Var b) = enCom a ++ enCom b
toArr n (Var a :@ y)     = enCom a ++ toU32 (n + 8) ++ toArr (n + 8) y
toArr n (x     :@ Var b) = toU32 (n + 8) ++ enCom b ++ toArr (n + 8) x
toArr n (x     :@ y)     = toU32 (n + 8) ++ toU32 nl ++ l ++ toArr nl y
  where l  = toArr (n + 8) x
        nl = n + 8 + length l
--neg32 n = toU32 $ 2^32 - n
neg32 n = [256 - n, 255, 255, 255]
toU32 = take 4 . byteMe
byteMe n | n < 256   = n : repeat 0
         | otherwise = n `mod` 256 : byteMe (n `div` 256)
\end{code}

== Machine Code ==

\begin{code}
br = 0xc
br_if = 0xd
getlocal = 0x20
setlocal = 0x21
teelocal = 0x22
i32load  = 0x28
i32store = 0x36
i32const = 0x41
i32ne    = 0x47
i32add   = 0x6a
i32sub   = 0x6b
i32mul   = 0x6c
i32shl   = 0x74
i32shr_s = 0x75
i32shr_u = 0x76
nPages = 8

leb128 n | n < 64   = [n]
         | n < 128  = [128 + n, 0]
         | otherwise = 128 + (n `mod` 128) : leb128 (n `div` 128)

varlen xs = leb128 $ length xs

lenc xs = varlen xs ++ xs

sect t xs = t : lenc (varlen xs ++ concat xs)

encStr s = lenc $ ord <$> s

encType "i32" = 0x7f
encType "f64" = 0x7c

encSig ins outs = 0x60  -- Function type.
  : lenc (encType <$> ins) ++ lenc (encType <$> outs)
\end{code}

Our binary starts the same as link:wasm.html[our first wasm demo], except we
work with `i32` instead of `f64` and ask for linear memory.

\begin{code}
compile e = concat [
  [0, 0x61, 0x73, 0x6d, 1, 0, 0, 0],  -- Magic string, version.
  -- Type section.
  sect 1 [encSig ["i32"] [], encSig [] [], encSig [] ["i32"]],
  -- Import section.
  sect 2 [
    -- [0, 0] = external_kind Function, type index 0.
    encStr "i" ++ encStr "f" ++ [0, 0],
    -- [0, 2] = external_kind Function, type index 2.
    encStr "i" ++ encStr "g" ++ [0, 2]],
  -- Function section.
  -- [1] = Type index.
  sect 3 [[1]],
  -- Memory section.
  -- 0 = no-maximum
  sect 5 [[0, nPages]],
  -- Export section.
  -- [0, 1] = external_kind Function, function index 2.
  sect 7 [encStr "e" ++ [0, 2]],
\end{code}

\begin{code}
  -- Code section.
  -- Locals
  let
    ip = 0  -- program counter
    sp = 1  -- stack pointer
    hp = 2  -- heap pointer
    ax = 3  -- accumulator
    bx = 4
  in sect 10 [lenc $ [1, 5, encType "i32",
    i32const] ++ varlen gen ++ [setlocal, ip,
    i32const] ++ leb128 (65536 * nPages) ++ [setlocal, sp,
    i32const] ++ varlen heap ++ [setlocal, hp,

    3, 0x40,  -- Lazy K loop
    -- BX = IP
    getlocal, ip, setlocal, bx,

    -- [HP] = IP
    getlocal, hp, getlocal, ip, i32store, 2, 0,
    -- [HP + 4] = Var "k"
    getlocal, hp, i32const, 4, i32add, i32const, 128 - 3, i32store, 2, 0,
    -- [HP + 8] = HP
    getlocal, hp, i32const, 8, i32add, getlocal, hp, i32store, 2, 0,
    -- [HP + 12] = Var "+"
    getlocal, hp, i32const, 12, i32add, i32const, 128 - 2, i32store, 2, 0,
    -- [HP + 16] = HP + 8
    getlocal, hp, i32const, 16, i32add, getlocal, hp, i32const, 8, i32add,
    i32store, 2, 0,
    -- [HP + 20] = Var "0"
    getlocal, hp, i32const, 20, i32add, i32const, 128 - 1, i32store, 2, 0,
    -- IP = HP + 16
    -- HP = HP + 24
    getlocal, hp, i32const, 16, i32add, setlocal, ip,
    getlocal, hp, i32const, 24, i32add, setlocal, hp,

    3, 0x40,  -- loop
    2, 0x40,  -- block 5
    2, 0x40,  -- block 4
    2, 0x40,  -- block 3
    2, 0x40,  -- block 2
    2, 0x40,  -- block 1
    2, 0x40,  -- block 0
    i32const, 128 - 1, getlocal, ip, i32sub,  -- -1 - IP
    0xe,5,0,1,2,3,4,5, -- br_table
    0xb,  -- end 0
-- Zero.

    2, 0x40,   -- block Z
    getlocal, ax, i32const, 128, 2, i32sub,  -- AX - 256
    br_if, 0,  -- br_if Z
    br, 8,     -- br function
    0xb,       -- end Z
    getlocal, ax, 0x10, 0,
    i32const, 0, setlocal, ax,

    -- [HP] = BX
    getlocal, hp, getlocal, bx, i32store, 2, 0,
    -- [HP + 4] = HP + 8
    getlocal, hp, i32const, 4, i32add, getlocal, hp, i32const, 8, i32add,
    i32store, 2, 0,
    -- [HP + 8] = Var "s"
    getlocal, hp, i32const, 8, i32add, i32const, 128 - 4, i32store, 2, 0,
    -- [HP + 12] = Var "k"
    getlocal, hp, i32const, 12, i32add, i32const, 128 - 3, i32store, 2, 0,
    -- IP = HP
    -- HP = HP + 16
    getlocal, hp, setlocal, ip,
    getlocal, hp, i32const, 16, i32add, setlocal, hp,
    br, 6,  -- br Lazy K loop

    0xb,  -- end 1
-- Successor.
    -- AX = AX + 1
    getlocal, ax, i32const, 1, i32add, setlocal, ax,
    -- IP = [[SP] + 4]
    getlocal, sp, i32load, 2, 0, -- align 2, offset 0.
    i32const, 4, i32add, i32load, 2, 0,
    setlocal, ip,
    -- SP = SP + 4
    -- In a correct program, the stack should now be empty.
    getlocal, sp, i32const, 4, i32add, setlocal, sp,
    br, 4,  -- br loop
    0xb,  -- end 2
-- K combinator.
    -- IP = [[SP] + 4]
    getlocal, sp, i32load, 2, 0, i32const, 4, i32add, i32load, 2, 0,
    setlocal, ip,
    -- SP = SP + 8
    getlocal, sp, i32const, 8, i32add, setlocal, sp,
    br, 3,  -- br loop
    0xb,  -- end 3
-- S combinator.
    -- [HP] = [[SP] + 4]
    getlocal, hp,
    getlocal, sp, i32load, 2, 0, i32const, 4, i32add, i32load, 2, 0,
    i32store, 2, 0,
    -- [HP + 4] = [[SP + 8] + 4]
    getlocal, hp, i32const, 4, i32add,
    getlocal, sp, i32const, 8, i32add, i32load, 2, 0,
    i32const, 4, i32add, i32load, 2, 0,
    i32store, 2, 0,
    -- [HP + 8] = [[SP + 4] + 4]
    getlocal, hp, i32const, 8, i32add,
    getlocal, sp, i32const, 4, i32add, i32load, 2, 0,
    i32const, 4, i32add, i32load, 2, 0,
    i32store, 2, 0,
    -- [HP + 12] = [HP + 4]
    getlocal, hp, i32const, 12, i32add,
    getlocal, hp, i32const, 4, i32add, i32load, 2, 0,
    i32store, 2, 0,
    -- SP = SP + 8
    -- [[SP]] = HP
    getlocal, sp, i32const, 8, i32add, teelocal, sp,
    i32load, 2, 0,
    getlocal, hp,
    i32store, 2, 0,
    -- [[SP] + 4] = HP + 8
    getlocal, sp, i32load, 2, 0, i32const, 4, i32add,
    getlocal, hp, i32const, 8, i32add,
    i32store, 2, 0,
    -- IP = HP
    -- HP = HP + 16
    getlocal, hp, teelocal, ip,
    i32const, 16, i32add, setlocal, hp,
    br, 2,  -- br loop
    0xb,  -- end 4
-- Input.
-- TODO: WRONG: reused nodes...
    -- [HP] = Vireo
    getlocal, hp, i32const] ++ leb128 addrVireo ++ [i32store, 2, 0,
    -- [HP + 4] = getChar * 8
    getlocal, hp, i32const, 4, i32add, 0x10, 1, i32const, 8, i32mul,
    i32store, 2, 0,
    -- [HP + 8] = HP
    getlocal, hp, i32const, 8, i32add, getlocal, hp, i32store, 2, 0,
    -- [HP + 12] = Var "<"
    getlocal, hp, i32const, 12, i32add, i32const, 128 - 5, i32store, 2, 0,
    -- IP = HP + 8
    getlocal, hp, i32const, 8, i32add, setlocal, ip,
    -- HP = HP + 16
    getlocal, hp, i32const, 16, i32add, setlocal, hp,
    br, 1,  -- br loop
    0xb,  -- end 5
-- Application.
    -- SP = SP - 4
    -- [SP] = IP
    getlocal, sp, i32const, 4, i32sub,
    teelocal, sp, getlocal, ip, i32store, 2, 0,

    -- If [IP] = Var "<" then getchar.
    2, 0x40,  -- block <
    getlocal, ip, i32load, 2, 0, i32const, 128 - 5, i32ne,
    br_if, 0,
    -- [HP] = Vireo
    getlocal, hp, i32const] ++ leb128 addrVireo ++ [i32store, 2, 0,
    -- [HP + 4] = getChar * 8
    getlocal, hp, i32const, 4, i32add, 0x10, 1, i32const, 8, i32mul,
    i32store, 2, 0,
    -- [HP + 8] = Var "<"
    getlocal, hp, i32const, 8, i32add, i32const, 128 - 5, i32store, 2, 0,
    -- [IP] = HP
    getlocal, ip, getlocal, hp, i32store, 2, 0,
    -- [IP + 4] = HP + 8
    getlocal, ip, i32const, 4, i32add, getlocal, hp, i32const, 8, i32add, i32store, 2, 0,
    -- HP = HP + 16
    getlocal, hp, i32const, 16, i32add, setlocal, hp,
    0xb,      -- end <

    -- IP = [IP]
    getlocal, ip, i32load, 2, 0, setlocal, ip,
    br, 0,
    0xb,    -- end loop

    0xb,    -- end Lazy K loop

    0xb]],  -- end function
\end{code}

The data section initializes the linear memory so our encoded tree sits
at the bottom.

\begin{code}
  -- Data section.
  sect 11 [[0, i32const, 0, 0xb] ++ lenc heap]]
  --where heap = encodeTree e
  where heap = encodeTree $ e :@ (Var "<" :@ Var "<")
  --where heap = encodeTree $ e :@ chuList "hello"
\end{code}

\begin{code}
#ifdef __HASTE__
dump asm = unwords $ xxShow <$> asm where
  xxShow c = reverse $ take 2 $ reverse $ '0' : showHex c ""

main = withElems ["input", "output", "sk", "asm", "evalB"] $
    \[iEl, oEl, skEl, aEl, evalB] -> do
  inp <- newMVar "hello"
  let
    putChar :: Int -> IO ()
    putChar c = do
      v <- getProp oEl "value"
      setProp oEl "value" $ v ++ " " ++ show c
    getChar :: IO Int
    getChar = do
      s <- takeMVar inp
      writeLog s
      case s of
        [] -> do
          putMVar inp []
          pure 256
        (h:t) -> do
          putMVar inp t
          pure $ ord h
  export "putChar" putChar
  export "getChar" getChar
  evalB `onEvent` Click $ const $ do
    setProp oEl "value" ""
    setProp skEl "value" ""
    setProp aEl "value" ""
    s <- getProp iEl "value"
    case parseProgram s of
      Left err -> setProp skEl "value" $ "error: " ++ show err
      Right sk -> do
        let asm = compile sk
        setProp skEl "value" $ show sk
        setProp aEl "value" $ show asm
        ffi "runWasmInts" asm :: IO ()
#endif
\end{code}
