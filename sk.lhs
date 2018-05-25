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
import Text.ParserCombinators.Parsec

infixl 5 :@
data Expr = Expr :@ Expr | Var String | Lam String Expr

source :: Parser [(String, Expr)]
source = catMaybes <$> many maybeLet where
  maybeLet = (((newline >>) . pure) =<<) . (ws >>) $
    optionMaybe $ (,) <$> v <*> (str "=" >> term)
  term = lam <|> app
  lam = flip (foldr Lam) <$> between lam0 lam1 (many1 v) <*> term where
    lam0 = str "\\" <|> str "\0955"
    lam1 = str "->" <|> str "."
  app = foldl1' (:@) <$> many1 ((Var <$> v) <|> between (str "(") (str ")") term)
  v   = many1 alphaNum >>= (ws >>) . pure
  str = (>> ws) . string
  ws = many (oneOf " \t") >> optional (try $ string "--" >> many (noneOf "\n"))
\end{code}

We won't directly translate a list of let expressions into assembly. Instead,
to make our job easier, we first translate to an intermediate language.

== What's in a name? ==

The names of variables in lambda abstractions only serve to record where a
variable is bound. The choice of a name is irrelevant, so long as we avoid
clashes.

This bookkeeping convention adds complexity. For example, we may need to rename
variables (&alpha;-conversion) to determine if two lambda terms are equal, or to avoid
names clobbering each other when building expressions.

One way to get rid of names is to use
https://en.wikipedia.org/wiki/De_Bruijn_index['De Bruijn indices]'. Briefly, we
replace each variable with an integer representing the number of `Lam` nodes we
encounter as we travel up the tree before reaching the binding abstraction. For
example,

\[
\lambda f.(\lambda x.x x)(\lambda x.f(x x))
\]

becomes:

\[
\lambda(\lambda 0 0)(\lambda 1(0 0))
\]

(Some prefer to start counting from 1 instead of 0.)

Instead of De Bruijn indices, we'll do something much more fun.

== SKI Combinator calculus ==

We define $S = \lambda x y z . x z (y z)$ and $K = \lambda x y . x$,
or in De Bruijn notation, $S = \lambda\lambda\lambda 2 0(1 0)$
and $K = \lambda\lambda 1$.
Then it turns out we can rewrite any closed lambda term with $S$ and $K$ alone.

First, we notice $S K K x = x$ for all $x$; a handy convention is to write $I$
for $S K K$. Then, we find all variables can be removed by recursively applying
the following:

\[
\begin{align}
\lceil \lambda x . x \rceil &= S K K \\
\lceil \lambda x . y \rceil &= K y \\
\lceil \lambda x . M N \rceil &= S \lceil \lambda x . M \rceil \lceil \lambda x . N \rceil
\end{align}
\]

where $\lceil T \rceil$ denotes the lambda term $T$ written without
lambda abstractions. This conversion is known as 'bracket
abstraction'. (In the third equation, $M, N$ denote lambda terms.)

Other choices of combinators are possible, for example, it turns out every
closed lambda term can be written using the
https://en.wikipedia.org/wiki/B,_C,_K,_W_system[B, C, K, W combinators] only.
We choose $S$ and $K$ so bracket abstraction is easy, and also so that we only
have to implement two functions to attain Turing completeness.

https://en.wikipedia.org/wiki/Iota_and_Jot[It's possible to combine $S$ and $K$
into one mega-combinator] (basically a Church-encoded pair) so the entire
program only uses a single combinator. We gain no advantage from doing this, at
least when it comes to writing a compiler.

Writing programs without variables is known as
https://en.wikipedia.org/wiki/Tacit_programming[tacit programming, or
point-free style].

== Better bracket abstraction ==

We refine the above rules to obtain leaner combinator calculus expressions.
One easy optimization is to generalize the second rule:
\[
\lceil \lambda x . M \rceil = K M \quad (x \notin M)
\]
which leads to the following code, where the `fv` function returns the free
variables of a given lambda term.

\begin{code}
fv vs (Var s) | s `elem` vs = []
              | otherwise   = [s]
fv vs (x :@ y)              = fv vs x `union` fv vs y
fv vs (Lam s f)             = fv (s:vs) f

babs0 env (Lam x e)
  | Var y <- t, x == y  = Var "s" :@ Var "k" :@ Var "k"
  | x `notElem` fv [] t = Var "k" :@ t
  | m :@ n <- t         = Var "s" :@
    babs0 env (Lam x m) :@ babs0 env (Lam x n)
  where t = babs0 env e
babs0 env (Var s)
  | Just t <- lookup s env = babs0 env t
  | otherwise              = Var s
babs0 env (m :@ n) = babs0 env m :@ babs0 env n
\end{code}

Even better are https://tromp.github.io/cl/LC.pdf[the rules
described by John Tromp]:

\begin{code}
babs env (Lam x e)
  | Var "s" :@ Var"k" :@ _ <- t = Var "s" :@ Var "k"
  | x `notElem` fv [] t = Var "k" :@ t
  | Var y <- t, x == y  = Var "s" :@  Var "k" :@ Var "k"
  | m :@ Var y <- t, x == y, x `notElem` fv [] m = m
  | Var y :@ m :@ Var z <- t, x == y, x == z =
    babs env $ Lam x $ Var "s" :@ Var "s" :@ Var "k" :@ Var x :@ m
  | m :@ (n :@ l) <- t, isComb m, isComb n =
    babs env $ Lam x $ Var "s" :@ Lam x m :@ n :@ l
  | (m :@ n) :@ l <- t, isComb m, isComb l =
    babs env $ Lam x $ Var "s" :@ m :@ Lam x l :@ n
  | (m :@ l) :@ (n :@ l') <- t, l `noLamEq` l', isComb m, isComb n
    = babs env $ Lam x $ Var "s" :@ m :@ n :@ l
  | m :@ n <- t        = Var "s" :@ babs env (Lam x m) :@ babs env (Lam x n)
  where t = babs env e
babs env (Var s)
  | Just t <- lookup s env = babs env t
  | otherwise              = Var s
babs env (m :@ n) = babs env m :@ babs env n

isComb e = null $ fv [] e \\ ["s", "k"]

noLamEq (Var x) (Var y) = x == y
noLamEq (a :@ b) (c :@ d) = a `noLamEq` c && b `noLamEq` d
noLamEq _ _ = False
\end{code}

Stricter checks such as avoiding recursive let definitions and prohibiting `s`
and `k` to be used as symbols are left as exercises.

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
integer `n`, the expression `Muz` evaluates to `u(u(...u(z)...))`, where there
are `n` occurrences of `u`, thus if `u` increments a counter (that is initially
zero) and `z` returns it, we have a routine that returns `n`.

== Graph Reduction ==

We encode the tree representing our program into an array, then
write WebAssembly to manipulate this tree. In other words, we model computation
as https://en.wikipedia.org/wiki/Graph_reduction['graph reduction'].

We view linear memory as an array of 32-bit integers. The values 0-3 represent
leaf nodes `z,u,k,s` in that order, while any other value `n` represents an
internal node with children represented by the 32-bit integers stored in linear
memory at `n` and `n + 4`.  We encode the tree so that address 4 holds the root
of the tree. Since 0 represents a leaf node, the first 4 bytes of linear memory
cannot be addressed, so their contents are initialized to zero and ignored.

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
encodeTree e = concatMap f $ 0 : toArr 4 e where
  f n | n < 4     = [n, 0, 0, 0]
      | otherwise = toU32 $ (n - 3) * 4
toU32 = take 4 . byteMe
byteMe n | n < 256   = n : repeat 0
         | otherwise = n `mod` 256 : byteMe (n `div` 256)
\end{code}

Because assembly is low-level, it helps to have a reference implementation of
the graph reduction function.

Our `run` function takes the address of a node to evaluate which we call `IP`,
a stack of addresses we've been previously asked to evaluate, and the current
state of linear memory.

For internal nodes, we push the current node on the stack then evaluate the
first child. For the `z` combinator, we return 0. For the `u` combinator we
return 1 plus the result of evaluating its argument, which we access via the
stack.  For the `k` combinator, we pop off the last two stack elements and
return the evaluation of its first argument.

Only `s` is tough to evaluate. In this case, we find the first free memory
address `b`, where we create two new internal nodes representing `xz` and `yz`,
where `x,y,z` are the arguments of `s`. Then we rewrite the immediate children
of the parent of the `z` node to point to the 2 newly created nodes. Lastly, we
pop 2 addresses off the stack and evaluate `b`. [I hope to eventually draw a
diagram to help explain this.]

We assume the input program is well-formed, that is, every `k` is given exactly
2 arguments, every `s` is given exactly 3 arguments, and so on.

\begin{code}
run p sp m = let
  get n = sum $ zipWith (*) ((m I.!) <$> [n..n+3]) ((256^) <$> [0..3])
  insList = foldr (\(k, a) m -> I.insert k a m)
  in case p of
    0 -> 0
    1 -> 1 + run (get $ head sp + 4) sp m
    2 -> run (get $ head sp + 4) (drop 2 sp) m
    3 -> run b (drop 2 sp) $ insList m $
        zip [b..]     (concatMap toU32 [x, z, y, z]) ++
        zip [sp!!2..] (concatMap toU32 [b, b + 8]) where
      b = I.size m
      [x, y, z] = get . (+4) <$> take 3 sp
    q -> run (get q) (p:sp) m
\end{code}

== Machine Code ==

Time to convert the above to assembly. We start with a few constants and
helpers:

\begin{code}
br = 0xc
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
  sect 1 [encSig ["i32"] [], encSig [] []],
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
statements break out a given number of blocks, a scheme reminiscent of De
Bruijn indices.

\begin{code}
  -- Code section.
  -- Locals
  let
    ip = 0  -- program counter
    sp = 1  -- stack pointer
    hp = 2  -- heap pointer
    ax = 3  -- accumulator
  in sect 10 [lenc $ [1, 4, encType "i32",
    i32const, 4, setlocal, ip,
    i32const] ++ leb128 (65536 * nPages) ++ [setlocal, sp,
    i32const] ++ varlen heap ++ [setlocal, hp,
    3, 0x40,  -- loop
    2, 0x40,  -- block 4
    2, 0x40,  -- block 3
    2, 0x40,  -- block 2
    2, 0x40,  -- block 1
    2, 0x40,  -- block 0
    getlocal, ip,
    0xe,4,0,1,2,3,4, -- br_table
    0xb,  -- end 0
-- Zero.
    getlocal, ax,
    0x10, 0,  -- call function 0
    br, 5,  -- br function
    0xb,  -- end 1
-- Successor.
    getlocal, ax, i32const, 1, i32add, setlocal, ax,
    -- IP = [[SP] + 4]
    getlocal, sp, i32load, 2, 0, -- align 2, offset 0.
    i32load, 2, 4,
    setlocal, ip,
    -- SP = SP + 4
    -- In a correct program, the stack should now be empty.
    getlocal, sp, i32const, 4, i32add, setlocal, sp,
    br, 3,  -- br loop
    0xb,  -- end 2
-- K combinator.
    -- IP = [[SP] + 4]
    getlocal, sp, i32load, 2, 0, i32load, 2, 4,
    setlocal, ip,
    -- SP = SP + 8
    getlocal, sp, i32const, 8, i32add, setlocal, sp,
    br, 2,  -- br loop
    0xb,  -- end 3
-- S combinator.
    -- [HP] = [[SP] + 4]
    getlocal, hp,
    getlocal, sp, i32load, 2, 0, i32load, 2, 4,
    i32store, 2, 0,
    -- [HP + 4] = [[SP + 8] + 4]
    getlocal, hp,
    getlocal, sp, i32load, 2, 8, i32load, 2, 4,
    i32store, 2, 4,
    -- [HP + 8] = [[SP + 4] + 4]
    getlocal, hp,
    getlocal, sp, i32load, 2, 4, i32load, 2, 4,
    i32store, 2, 8,
    -- [HP + 12] = [HP + 4]
    getlocal, hp,
    getlocal, hp, i32load, 2, 4,
    i32store, 2, 12,
    -- SP = SP + 8
    -- [[SP]] = HP
    getlocal, sp, i32const, 8, i32add, teelocal, sp,
    i32load, 2, 0,
    getlocal, hp,
    i32store, 2, 0,
    -- [[SP] + 4] = HP + 8
    getlocal, sp, i32load, 2, 0,
    getlocal, hp, i32const, 8, i32add,
    i32store, 2, 4,
    -- IP = HP
    -- HP = HP + 16
    getlocal, hp, teelocal, ip,
    i32const, 16, i32add, setlocal, hp,
    br, 1,  -- br loop
    0xb,  -- end 4
-- Application.
    -- SP = SP - 4
    -- [SP] = IP
    getlocal, sp, i32const, 4, i32sub,
    teelocal, sp, getlocal, ip, i32store, 2, 0,
    -- IP = [IP]
    getlocal, ip, i32load, 2, 0, setlocal, ip,
    br, 0,
    0xb,    -- end loop
    0xb]],  -- end function
\end{code}

The data section initializes the linear memory so our encoded tree sits
at the bottom.

\begin{code}
  -- Data section.
  sect 11 [[0, i32const, 0, 0xb] ++ lenc heap]]
  where heap = encodeTree e
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
    setResult d = setProp oEl "value" $ show d
  export "setResult" setResult
  evalB `onEvent` Click $ const $ do
    setProp oEl "value" ""
    setProp skEl "value" ""
    setProp aEl "value" ""
    s <- getProp iEl "value"
    case toSK s of
      Left err -> setProp skEl "value" $ "error: " ++ show err
      Right sk -> do
        let asm = compile sk
        setProp skEl "value" $ showSK sk
        setProp aEl "value" $ dump asm
        ffi "runWasmInts" asm :: IO ()
#else
main = interact $ \s -> case toSK s of
  Left err -> "error: " ++ show err
  Right sk -> unlines [
    showSK sk,
    show $ compile sk,
    show $ run 4 [] $ I.fromAscList $ zip [0..] $ encodeTree sk]
#endif
\end{code}

During development, I was helped by a REPL for the intermediate language:

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
      outputStrLn $ show $ compile e
      outputStrLn $ show $ run 4 [] $ I.fromAscList $ zip [0..] $ encodeTree e
      skRepl
#endif
\end{code}
