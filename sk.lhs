= SK =

[pass]
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
<script src="sk.js"></script>
<p><textarea style="border: solid 4px; border-color: #999999" id="input" rows="1" cols="40">1/(sqrt 8/9801*1103)</textarea>
<br>
<span id="output"></span>
<br>
<button id="evalB">Compile & Run</button>
<br>
<p>
<b>wasm</b>:
<br>
<textarea id="asm" rows="12" cols="25" readonly></textarea></p>
<script type="text/javascript">
function runWasmInts(a) {
  WebAssembly.instantiate(new Uint8Array(a),
    {i:{f:x => console.log(x)}}).then(x => x.instance.exports.e());
}
runWasmInts([0,97,115,109,1,0,0,0,1,8,2,96,1,127,0,96,0,0,2,7,1,1,105,1,102,0,0,3,2,1,1,5,3,1,0,1,7,5,1,1,101,0,1,10,245,1,1,242,1,1,4,127,65,4,33,0,65,128,128,4,33,1,65,188,1,33,3,3,64,2,64,2,64,2,64,2,64,2,64,32,0,14,4,0,1,2,3,4,11,32,2,16,0,12,5,11,32,2,65,1,106,33,2,32,1,40,2,0,65,4,106,40,2,0,33,0,32,1,65,4,106,33,1,12,3,11,32,1,40,2,0,65,4,106,40,2,0,33,0,32,1,65,8,106,33,1,12,2,11,32,3,32,1,40,2,0,65,4,106,40,2,0,54,2,0,32,3,65,4,106,32,1,65,8,106,40,2,0,65,4,106,40,2,0,54,2,0,32,3,65,8,106,32,1,65,4,106,40,2,0,65,4,106,40,2,0,54,2,0,32,3,65,12,106,32,3,65,4,106,40,2,0,54,2,0,32,1,65,8,106,34,1,40,2,0,32,3,54,2,0,32,1,40,2,0,65,4,106,32,3,65,8,106,54,2,0,32,3,34,0,65,16,106,33,3,12,1,11,32,1,65,4,107,34,1,32,0,54,2,0,32,0,40,2,0,33,0,12,0,11,11,11,195,1,1,0,65,0,11,188,1,0,0,0,0,12,0,0,0,0,0,0,0,20,0,0,0,1,0,0,0,28,0,0,0,76,0,0,0,36,0,0,0,60,0,0,0,3,0,0,0,44,0,0,0,52,0,0,0,2,0,0,0,3,0,0,0,2,0,0,0,68,0,0,0,2,0,0,0,3,0,0,0,2,0,0,0,84,0,0,0,132,0,0,0,92,0,0,0,116,0,0,0,3,0,0,0,100,0,0,0,108,0,0,0,2,0,0,0,3,0,0,0,2,0,0,0,124,0,0,0,2,0,0,0,3,0,0,0,2,0,0,0,140,0,0,0,172,0,0,0,3,0,0,0,148,0,0,0,156,0,0,0,2,0,0,0,3,0,0,0,164,0,0,0,2,0,0,0,3,0,0,0,180,0,0,0,2,0,0,0,3,0,0,0,2,0,0,0]);
</script>
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

4:  s(skk)(skk)(s(s(ks)k)(skk))uz
256:  s(skk)(skk)(s(skk)(skk)(s(s(ks)k)(skk)))uz
SII(SII(S(S(KS)K)I))

\begin{code}
import Data.Char
import qualified Data.IntMap as I
import Data.List
import Text.ParserCombinators.Parsec
import System.Console.Readline

data Expr = App Expr Expr | Var String deriving Show

expr :: Parser Expr
expr = foldl1 App <$> many1 ((Var . pure <$> letter) <|> between (char '(') (char ')') expr)

main = repl

repl = do
  ms <- readline "> "
  case ms of
    Nothing -> putStrLn ""
    Just s  -> do
      let Right e = parse expr "" s
      print $ compile e
      print $ concatMap to4 $ 0 : toArr 4 e
      print e
      print $ run 4 [] $ I.fromAscList $ zip [0..] $ concatMap to4 $ 0 : toArr 4 e
      repl

toArr n (Var "s") = [3]
toArr n (Var "k") = [2]
toArr n (Var "u") = [1]
toArr n (Var "z") = [0]
{-
toArr n (App x y)
  | length l == 1 = l ++ encR []
  | otherwise     = n + 2 : encR l
  where
    l = toArr (n + 2) x
    encR x | length r == 1 = r ++ x
           | otherwise     = m : x ++ r
           where
             m = n + 2 + length x
             r = toArr m y
-}
toArr n (App x@(Var _) y@(Var _)) = toArr n x ++ toArr n y
toArr n (App x@(Var _) y) = toArr n x ++ [n + 2] ++ toArr (n + 2) y
toArr n (App x y@(Var _)) = n + 2 : toArr n y ++ toArr (n + 2) x
toArr n (App x y) = [n + 2, nl] ++ l ++ toArr nl y where
  l = toArr (n + 2) x
  nl = n + 2 + length l

run p sp m = let
  get n = m I.! n
  insList m = foldr (\(k, a) m -> I.insert k a m) m
  in case p of
    0 -> 0
    1 -> 1 + run (get $ head sp + 4) [] m
    2 -> run (get $ head sp + 4) (drop 2 sp) m
    -- 3 -> run b (drop 3 sp) $ insList $ zip [b..] $ (concatMap (:[0,0,0]) [b + 8, b + 16, x, z, y, z]) where
    3 -> run b (drop 2 sp) $ insList m $ zip [b..] (concatMap (:[0,0,0]) [x, z, y, z]) ++ zip [sp!!2..] (concatMap (:[0,0,0]) [b, b + 8]) where
      b = I.size m
      [x, y, z] = get . (+4) <$> take 3 sp
    q -> run (get q) (p:sp) m
\end{code}

\begin{code}
br = 0xc
i32load = 0x28
i32store = 0x36
i32const = 0x41
i32add = 0x6a
i32sub = 0x6b
i32mul = 0x6c
i32shl   = 0x74
i32shr_s = 0x75
i32shr_u = 0x76
getlocal = 0x20
setlocal = 0x21
teelocal = 0x22

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
  -- [0, 1] = no-maximum, 1 page.
  sect 5 [[0, 1]],
  -- Export section.
  -- [0, 1] = external_kind Function, index 1.
  sect 7 [encStr "e" ++ [0, 1]],
  -- Code section.
  sect 10 [lenc $ [1, 4, encType "i32",
    -- L0 = program counter
    -- L1 = stack pointer
    -- L2 = result
    -- L3 = heap
    i32const, 4, setlocal, 0,
    i32const] ++ leb128 65536 ++ [setlocal, 1,
    i32const] ++ varlen heap ++ [setlocal, 3,
    3, 0x40,  -- loop
    2, 0x40,
    2, 0x40,
    2, 0x40,
    2, 0x40,
    2, 0x40,
    getlocal, 0,
    0xe,4,0,1,2,3,4, -- br_table
    0xb,  -- end 0
-- Zero.
    getlocal, 2,
    0x10, 0,
    br, 5,
    0xb,  -- end 1
-- Successor.
    getlocal, 2, i32const, 1, i32add, setlocal, 2,
    -- IP = [[SP] + 4]
    getlocal, 1,
    i32load, 2, 0, -- align 2, offset 0.
    i32const, 4, i32add,
    i32load, 2, 0, -- align 2, offset 0.
    setlocal, 0,
    -- SP = SP + 4
    -- In a correct program, the stack should now be empty.
    getlocal, 1, i32const, 4, i32add, setlocal, 1,
    br, 3,  -- br loop
    0xb,  -- end 2
-- K combinator.
    -- IP = [[SP] + 4]
    getlocal, 1,
    i32load, 2, 0, -- align 2, offset 0.
    i32const, 4, i32add,
    i32load, 2, 0, -- align 2, offset 0.
    setlocal, 0,
    -- SP = SP + 8
    getlocal, 1, i32const, 8, i32add, setlocal, 1,
    br, 2,  -- br loop
\end{code}

\begin{code}
    0xb,  -- end 3
-- S combinator.
    -- [HP] = [[SP] + 4]
    getlocal, 3,
    getlocal, 1, i32load, 2, 0, i32const, 4, i32add, i32load, 2, 0,
    i32store, 2, 0,
    -- [HP + 4] = [[SP + 8] + 4]
    getlocal, 3, i32const, 4, i32add,
    getlocal, 1, i32const, 8, i32add, i32load, 2, 0, i32const, 4, i32add, i32load, 2, 0,
    i32store, 2, 0,
    -- [HP + 8] = [[SP + 4] + 4]
    getlocal, 3, i32const, 8, i32add,
    getlocal, 1, i32const, 4, i32add, i32load, 2, 0, i32const, 4, i32add, i32load, 2, 0,
    i32store, 2, 0,
    -- [HP + 12] = [HP + 4]
    getlocal, 3, i32const, 12, i32add,
    getlocal, 3, i32const, 4, i32add, i32load, 2, 0,
    i32store, 2, 0,
    -- SP = SP + 8
    -- [[SP]] = HP
    getlocal, 1, i32const, 8, i32add, teelocal, 1,
    i32load, 2, 0,
    getlocal, 3,
    i32store, 2, 0,
    -- [[SP] + 4] = HP + 8
    getlocal, 1, i32load, 2, 0, i32const, 4, i32add,
    getlocal, 3, i32const, 8, i32add,
    i32store, 2, 0,
    -- IP = HP
    -- HP = HP + 16
    getlocal, 3, teelocal, 0,
    i32const, 16, i32add, setlocal, 3,
{-
    -- [HP] = HP + 8
    getlocal, 3,
    getlocal, 3, i32const, 8, i32add,
    i32store, 2, 0,
    -- [HP + 4] = HP + 16
    getlocal, 3, i32const, 4, i32add,
    getlocal, 3, i32const, 16, i32add,
    i32store, 2, 0,
    -- [HP + 8] = [[SP] + 4]
    getlocal, 3, i32const, 8, i32add,
    getlocal, 1, i32load, 2, 0, i32const, 4, i32add, i32load, 2, 0,
    i32store, 2, 0,
    -- [HP + 12] = [[SP + 8] + 4]
    getlocal, 3, i32const, 12, i32add,
    getlocal, 1, i32const, 8, i32add, i32load, 2, 0, i32const, 4, i32add, i32load, 2, 0,
    i32store, 2, 0,
    -- [HP + 16] = [[SP + 4] + 4]
    getlocal, 3, i32const, 16, i32add,
    getlocal, 1, i32const, 4, i32add, i32load, 2, 0, i32const, 4, i32add, i32load, 2, 0,
    i32store, 2, 0,
    -- [HP + 20] = [HP + 12]
    getlocal, 3, i32const, 20, i32add,
    getlocal, 3, i32const, 12, i32add, i32load, 2, 0,
    i32store, 2, 0,
    -- IP = HP
    -- HP = HP + 24
    getlocal, 3, teelocal, 0,
    i32const, 24, i32add, setlocal, 3,
    -- SP = SP + 12
    getlocal, 1, i32const, 12, i32add, setlocal, 1,
-}
    br, 1,  -- br loop
    0xb,  -- end 4
-- Application.
    -- SP = SP - 4
    -- [SP] = IP
    getlocal, 1, i32const, 4, i32sub,
    teelocal, 1, getlocal, 0, i32store, 2, 0,
    -- IP = [IP]
    getlocal, 0, i32load, 2, 0, setlocal, 0,
    br, 0,
    0xb,    -- end loop
    0xb]],  -- end function

  -- Data section.
  sect 11 $ [[0, i32const, 0, 0xb] ++ lenc heap]]
  where heap = concatMap to4 $ 0 : toArr 4 e

to4 n | n < 4 = [n, 0, 0, 0]
      | otherwise = take 4 $ f $ (n - 3) * 4
      where
        f n | n < 256   = n : repeat 0
            | otherwise = n `mod` 256 : f (n `div` 256)

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

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
<script type="text/javascript">
/*
var sheet = document.createElement('style');
sheet.innerHTML = "body { background-color: black; color: #757575;} pre { background-color: black; }"
document.body.appendChild(sheet);
*/
</script>
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
