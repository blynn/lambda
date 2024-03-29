= A Combinatory Compiler =

The compiler below accepts a Turing-complete language and produces WebAssembly.
The source should consist of lambda calculus definitions including a function
`main` that outputs a Church-encoded integer.

[pass]
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
<script src="sk.js"></script>
<textarea id="input" rows="16" cols="80">
true = \x y -> x
false = \x y -> y
0 = \f x -> x
1 = \f x -> f x
succ = \n f x -> f(n f x)
pred = \n f x -> n(\g h -> h (g f)) (\u -> x) (\u ->u)
mul = \m n f -> m(n f)
is0 = \n -> n (\x -> false) true
Y = \f -> (\x -> x x)(\x -> f(x x))
fact = Y(\f n -> (is0 n) 1 (mul n (f (pred n))))
main = fact (succ (succ (succ 1)))  -- Compute 4!
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
    {i:{f:x => Haste.setResult(x)}}).then(x => x.instance.exports.e());
}
</script>
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

== Parser ==

We build off link:index.html[our lambda calculus parser]:

\begin{code}
{-# LANGUAGE CPP #-}
#ifdef __HASTE__
{-# LANGUAGE OverloadedStrings #-}
import Haste.DOM
import Haste.Events
import Haste.Foreign
import Numeric
#else
import System.Console.Haskeline
#endif
import Data.Char
import qualified Data.IntMap as I
import Data.List
import Data.Maybe
import Text.Parsec

infixl 5 :@
data Expr = Expr :@ Expr | Var String | Lam String Expr deriving Eq

source :: Parsec String () [(String, Expr)]
source = catMaybes <$> many maybeLet where
  maybeLet = between ws newline $ optionMaybe $ (,) <$> v <*> (str "=" >> term)
  term = lam <|> app
  lam = flip (foldr Lam) <$> between lam0 lam1 (many1 v) <*> term where
    lam0 = str "\\" <|> str "\955"
    lam1 = str "->" <|> str "."
  app = foldl1' (:@) <$> many1
    ((Var <$> v) <|> between (str "(") (str ")") term)
  v   = many1 alphaNum <* ws
  str = (>> ws) . string
  ws = many (oneOf " \t") >> optional (try $ string "--" >> many (noneOf "\n"))
\end{code}

== Combinators ==

We recap part of link:cl.html[our notes on combinatory logic]. Define the
combinators:

  * $S = \lambda x y z . x z (y z)$

  * $K = \lambda x y .  x$

The classic bracket abstraction algorithm with the K-optimization is:

\[
\begin{align}
\lceil \lambda x . M \rceil &= K M \quad (x \notin M) \\
\lceil \lambda x . x \rceil &= S K K \\
\lceil \lambda x . M N \rceil &= S \lceil \lambda x . M \rceil \lceil \lambda x . N \rceil
\end{align}
\]

Any closed lambda term can be rewritten in terms of $S$ and $K$ by applying the
above rules starting from the innermost lambda abstraction and working
outwards:

\begin{code}
lacks x t = case t of
  Var s | s == x -> False
  u :@ v -> lacks x u && lacks x v
  _ -> True

babs0 env (Lam x e)
  | lacks x t   = Var "k" :@ t
  | otherwise   = case t of
    Var y  -> Var "s" :@ Var "k" :@ Var "k"
    m :@ n -> Var "s" :@ babs0 env (Lam x m) :@ babs0 env (Lam x n)
  where t = babs0 env e
babs0 env (Var s)
  | Just t <- lookup s env = babs0 env t
  | otherwise              = Var s
babs0 env (m :@ n) = babs0 env m :@ babs0 env n
\end{code}

We also mentioned David Turner found more optimizations, enough to make bracket
abstraction practical. However, he used more combinators than just $S$ and $K$.
Luckily, https://tromp.github.io/cl/LC.pdf[John Tromp ported the rules to $S$
and $K$]:

\begin{code}
babs env (Lam x e) = go $ babs env e where
  go t
    | Var "s" :@ Var "k" :@ _ <- t = Var "s" :@ Var "k"
    | lacks x t = Var "k" :@ t
    | Var y <- t, x == y  = Var "s" :@  Var "k" :@ Var "k"
    | m :@ Var y <- t, x == y, lacks x m = m
    | Var y :@ m :@ Var z <- t, x == y, x == z =
      go $ Var "s" :@ Var "s" :@ Var "k" :@ Var x :@ m
    | m :@ (n :@ l) <- t, isComb m, isComb n =
      go $ Var "s" :@ go m :@ n :@ l
    | (m :@ n) :@ l <- t, isComb m, isComb l =
      go $ Var "s" :@ m :@ go l :@ n
    | (m :@ l) :@ (n :@ l') <- t, l == l', isComb m, isComb n =
      go $ Var "s" :@ m :@ n :@ l
    | m :@ n <- t = Var "s" :@ go m :@ go n
babs env (Var s)
  | Just t <- lookup s env = babs env t
  | otherwise              = Var s
babs env (m :@ n) = babs env m :@ babs env n

isComb t = case t of
  Var "s" -> True
  Var "k" -> True
  Var _ -> False
  u :@ v -> isComb u && isComb v
\end{code}

The above assumes we have no recursive let definitions and that `s` and `k`
are reserved keywords. Enforcing this is left as an exercise.

A few lines in the Either monad glues together our parser and our bracket
abstraction routine:

\begin{code}
toSK s = do
  env <- parse source "" (s ++ "\n")
  case lookup "main" env of
    Nothing -> Left $ error "missing main"
    Just t -> pure $ babs env t :@ Var "u" :@ Var "z"
\end{code}

We've introduced two more combinators: `u` and `z`, which we think of as the
successor function and zero respectively. Given a Church encoding `M` of an
integer `n`, the expression `Muz` evaluates to `u(u(...u(z)...))`, where
there are `n` occurrences of `u`. We make `u` increment a counter, and we
make `z` return it, so when evaluated in normal order it returns `n`.

== Graph Reduction ==

We encode the tree representing our program into an array, then write
WebAssembly to manipulate this tree. In other words, we model computation
as https://en.wikipedia.org/wiki/Graph_reduction['graph reduction'].

We view linear memory as an array of 32-bit integers. The values 0-3
represent leaf nodes `z,u,k,s` in that order, while any other value `n`
represents an internal node with children represented by the 32-bit integers
stored in linear memory at `n` and `n + 4`.

We encode the tree so that address 4 holds the root of the tree. Since 0
represents a leaf node, the first 4 bytes of linear memory cannot be
addressed, so their contents are initialized to zero and ignored.

\begin{code}
toArr n (Var "z") = [0]
toArr n (Var "u") = [1]
toArr n (Var "k") = [2]
toArr n (Var "s") = [3]
toArr n (x@(Var _) :@ y@(Var _)) = toArr n x ++ toArr n y
toArr n (x@(Var _) :@ y)         = toArr n x ++ [n + 2] ++ toArr (n + 2) y
toArr n (x         :@ y@(Var _)) = n + 2 : toArr n y ++ toArr (n + 2) x
toArr n (x         :@ y)         = [n + 2, nl] ++ l ++ toArr nl y
  where l  = toArr (n + 2) x
        nl = n + 2 + length l
encodeTree :: Expr -> [Int]
encodeTree e = concatMap f $ 0 : toArr 4 e where
  f n | n < 4     = [n, 0, 0, 0]
      | otherwise = toU32 $ (n - 3) * 4
toU32 = take 4 . byteMe
byteMe n | n < 256   = n : repeat 0
         | otherwise = n `mod` 256 : byteMe (n `div` 256)
\end{code}

Our `run` function takes the current and a stack of addresses state of linear
memory, and simulates what our assembly code will do.

For the `z` combinator, we return 0. For the `u` combinator we return 1 plus
the result of evaluating its argument.
For the `k` combinator, we pop off the last two stack elements and push the
evaluation of its first argument.

For `s` we create two internal nodes representing `xz` and `yz` on the the
heap `hp`, where `x,y,z` are the arguments of `s`. Then we lazily evaluate:
we rewrite the immediate children of the parent of the `z` node to apply the
first of the newly created nodes to the other.

For internal nodes, we push the first child on the stack then recurse.

We assume the input program is well-formed, that is, every `k` is given
exactly 2 arguments, every `s` is given exactly 3 arguments, and so on.

\begin{code}
run m (p:sp) = case p of
  0 -> 0
  1 -> 1 + run m (arg 0 : sp)
  2 -> run m $ arg 0 : drop 2 sp
  3 -> run m' $ hp:drop 2 sp where
    m' = insList m $
      zip [hp..]    (concatMap toU32 [arg 0, arg 2, arg 1, arg 2]) ++
      zip [sp!!2..] (concatMap toU32 [hp, hp + 8])
    hp = I.size m
  _ -> run m $ get p:p:sp
  where
  arg k = get (sp!!k + 4)
  get n = sum $ zipWith (*) ((m I.!) <$> [n..n+3]) ((256^) <$> [0..3])
  insList = foldr (\(k, a) m -> I.insert k a m)
\end{code}

== Machine Code ==

We convert the above to assembly. First, a few constants and helpers:

\begin{code}
compile :: [Int] -> [Int]
compile heap = let
  typeFunc = 0x60
  typeI32  = 0x7f
  br       = 0xc
  getlocal = 0x20
  setlocal = 0x21
  teelocal = 0x22
  i32load  = 0x28
  i32store = 0x36
  i32const = 0x41
  i32add   = 0x6a
  i32sub   = 0x6b
  i32mul   = 0x6c
  i32shl   = 0x74
  i32shr_s = 0x75
  i32shr_u = 0x76
  i64const = 0x42
  i64store = 0x37
  i64shl   = 0x86
  i64add   = 0x7c
  i64load32u    = 0x35
  i64extendui32 = 0xac
  nPages = 8
  leb128 n | n < 64   = [n]
           | n < 128  = [128 + n, 0]
           | otherwise = 128 + (n `mod` 128) : leb128 (n `div` 128)
  varlen xs = leb128 $ length xs
  lenc xs = varlen xs ++ xs
  encStr s = lenc $ ord <$> s
  encSig ins outs = typeFunc : lenc ins ++ lenc outs
  sect t xs = t : lenc (varlen xs ++ concat xs)
\end{code}

We link:../asm/wasm.html[follow the wasm format]:

\begin{code}
  in concat [
  [0, 0x61, 0x73, 0x6d, 1, 0, 0, 0],  -- Magic string, version.
  -- Type section.
  sect 1 [encSig [typeI32] [], encSig [] []],
  -- Import section.
  -- [0, 0] = external_kind Function, index 0.
  sect 2 [encStr "i" ++ encStr "f" ++ [0, 0]],
  -- Function section.
  -- [1] = Type index.
  sect 3 [[1]],
  -- Memory section.
  -- 0 = no-maximum
  sect 5 [[0, nPages]],
  -- Export section.
  -- [0, 1] = external_kind Function, index 1.
  sect 7 [encStr "e" ++ [0, 1]],
\end{code}

We compile the `run` function by hand. Initially, our tree is encoded at the
bottom of the linear memory, and the stack pointer is at the top.

We encounter features of WebAssembly may surprise those who
accustomed to other instruction sets.

Load and store instructions must be given alignment and offset arguments.

There are no explicit labels or jumps. Instead, labels are implicitly defined
by declaring well-nested `block-end` and `loop-end` blocks, and branch
statements break out a given number of blocks.

\begin{code}
  -- Code section.
  -- Locals
  let
    sp = 0  -- stack pointer
    hp = 1  -- heap pointer
    ax = 2  -- accumulator
  in sect 10 [lenc $ [1, 3, typeI32,
    -- SP = 65536 * nPages - 4
    -- [SP] = 4
    i32const] ++ leb128 (65536 * nPages - 4) ++ [teelocal, sp,
    i32const, 4, i32store, 2, 0,
    i32const] ++ varlen heap ++ [setlocal, hp,
    3, 0x40,  -- loop
    2, 0x40,  -- block 4
    2, 0x40,  -- block 3
    2, 0x40,  -- block 2
    2, 0x40,  -- block 1
    2, 0x40,  -- block 0
    getlocal, sp, i32load, 2, 0,
    0xe,4,0,1,2,3,4, -- br_table
    0xb,  -- end 0
-- Zero.
    getlocal, ax, 0x10, 0,  -- call function 0
    br, 5,  -- br function
    0xb,  -- end 1
-- Successor.
    getlocal, ax, i32const, 1, i32add, setlocal, ax,
    -- SP = SP + 4
    -- [SP] = [[SP] + 4]
    getlocal, sp, i32const, 4, i32add, teelocal, sp,
    getlocal, sp, i32load, 2, 0, i32load, 2, 4, i32store, 2, 0,
    br, 3,  -- br loop
    0xb,  -- end 2
-- K combinator.
    -- [SP + 8] = [[SP + 4] + 4]
    getlocal, sp,
    getlocal, sp, i32load, 2, 4, i32load, 2, 4,
    i32store, 2, 8,
    -- SP = SP + 8
    getlocal, sp, i32const, 8, i32add, setlocal, sp,
    br, 2,  -- br loop
    0xb,  -- end 3
-- S combinator.
    -- [HP] = [[SP + 4] + 4]
    -- [HP + 4] = [[SP + 12] + 4]
    getlocal, hp,
    getlocal, sp, i32load, 2, 4, i64load32u, 2, 4,
    getlocal, sp, i32load, 2, 12, i64load32u, 2, 4,
    i64const, 32, i64shl, i64add, i64store, 3, 0,
    -- [HP + 8] = [[SP + 8] + 4]
    -- [HP + 12] = [HP + 4]
    getlocal, hp,
    getlocal, sp, i32load, 2, 8, i64load32u, 2, 4,
    getlocal, hp, i64load32u, 2, 4,
    i64const, 32, i64shl, i64add, i64store, 3, 8,
    -- SP = SP + 12
    -- [[SP]] = HP
    -- [[SP] + 4] = HP + 8
    getlocal, sp, i32const, 12, i32add, teelocal, sp,
    i32load, 2, 0,
    getlocal, hp, i64extendui32,
    getlocal, hp, i32const, 8, i32add,
    i64extendui32, i64const, 32, i64shl, i64add, i64store, 3, 0,
    -- HP = HP + 16
    getlocal, hp, i32const, 16, i32add, setlocal, hp,
    br, 1,  -- br loop
    0xb,  -- end 4
-- Application.
    -- SP = SP - 4
    -- [SP] = [[SP + 4]]
    getlocal, sp, i32const, 4, i32sub,
    teelocal, sp, getlocal, sp, i32load, 2, 4, i32load, 2, 0, i32store, 2, 0,
    br, 0,
    0xb,    -- end loop
    0xb]],  -- end function
\end{code}

The data section initializes the linear memory so our encoded tree sits
at the bottom.

\begin{code}
  -- Data section.
  sect 11 [[0, i32const, 0, 0xb] ++ lenc heap]]
\end{code}

To keep the code simple, we ignore garbage collection. Because we represent
numbers in unary, and also because we only ask for a few pages of memory,
our demo only works on relatively small programs.

== User Interface ==

For the demo, we add a couple of helpers to show the intermediate form and
assembly opcodes.

\begin{code}
showSK (Var s)  = s
showSK (x :@ y) = showSK x ++ showR y where
  showR (Var s) = s
  showR _       = "(" ++ showSK y ++ ")"

#ifdef __HASTE__
dump asm = unwords $ xxShow <$> asm where
  xxShow c = reverse $ take 2 $ reverse $ '0' : showHex c ""

main = withElems ["input", "output", "sk", "asm", "evalB"] $
    \[iEl, oEl, skEl, aEl, evalB] -> do
  let
    setResult :: Int -> IO ()
    setResult = setProp oEl "value" . show
  export "setResult" setResult
  evalB `onEvent` Click $ const $ do
    setProp oEl "value" ""
    setProp skEl "value" ""
    setProp aEl "value" ""
    s <- getProp iEl "value"
    case toSK s of
      Left err -> setProp skEl "value" $ "error: " ++ show err
      Right sk -> do
        let asm = compile $ encodeTree sk
        setProp skEl "value" $ showSK sk
        setProp aEl "value" $ dump asm
        ffi "runWasmInts" asm :: IO ()
#else
main = interact $ \s -> case toSK s of
  Left err -> "error: " ++ show err
  Right sk -> unlines
    [ showSK sk
    , show $ compile $ encodeTree sk
    , show $ run (I.fromAscList $ zip [0..] $ encodeTree sk) [4]
    ]
#endif
\end{code}

During development, a REPL for the intermediate language was helpful:

\begin{code}
#ifndef __HASTE__
expr :: Parser Expr
expr = foldl1 (:@) <$>
  many1 ((Var . pure <$> letter) <|> between (char '(') (char ')') expr)

skRepl :: InputT IO ()
skRepl = do
  ms <- getInputLine "> "
  case ms of
    Nothing -> outputStrLn ""
    Just s  -> do
      let Right e = parse expr "" s
      outputStrLn $ show $ encodeTree e
      outputStrLn $ show $ compile $ encodeTree e
      outputStrLn $ show $ run (I.fromAscList $ zip [0..] $ encodeTree e) [4]
      skRepl
#endif
\end{code}
