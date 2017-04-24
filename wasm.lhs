= WebAssembly =

Click below to compile an expression to
http://webassembly.org/[WebAssembly] and run it.

[pass]
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
<script src="wasm.js"></script>
<p><textarea style="border: solid 4px; border-color: #999999" id="input" rows="1" cols="40">1/(sqrt 8/9801*1103)</textarea>
<br>
<span id="output"></span>
<br>
<button id="evalB">Compile + Run</button>
<br>
<p>
<b>wasm</b>:
<br>
<textarea id="asm" rows="12" cols="25" readonly></textarea></p>
<script type="text/javascript">
function runWasmInts(a) {
  WebAssembly.instantiate(new Uint8Array(a),
    {i:{f:x => Haste.setResult(x)}}).then(x => x.instance.exports.e());
}
</script>
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Available operators:

  abs neg ceil floor trunc nearest sqrt + - * / min max copysign

== The wasm binary format ==

We open with the 4-byte magic string `"\0asm"`, then the 4-byte version,
which is 1 in little-endian:

------------------------------------------------------------------------------
00 61 73 6d 01 00 00 00
------------------------------------------------------------------------------

Apart from this version number, WebAssembly encodes integers with
https://en.wikipedia.org/wiki/LEB128[LEB128], using the signed variant when
appropriate. We shall refer to the encoded numbers as varuints and varints.

The numbers in our example are small enough that a varint or a varuint can be
thought of as a byte (holding a number between 0 and 127, or -64 and 63).

A number of sections follow the first 8 bytes. Each section begins with a
varuint for the section ID, followed by a varuint for the length of the
section.

Section 1 declares the types of function signatures.
We'll export a function `e` for JavaScript to call, and import a function `f`
from the import object `i` through which it returns a 32-bit integer.

The type signature of the imported function is:

------------------------------------------------------------------------------
60  ; Function.
01  ; One input.
7f  ; 32-bit integer (i32).
00  ; No outputs.
------------------------------------------------------------------------------

and the type signature of the exported function is:

------------------------------------------------------------------------------
60  ; Function.
00  ; No inputs.
00  ; No outputs.
------------------------------------------------------------------------------

Thus the entire type section is:

------------------------------------------------------------------------------
01 08  ; Type section follows. It's 8 bytes long.
02     ; Two type signatures follow.
60 01 7f 00 60 00 00
------------------------------------------------------------------------------

Section 2 declares imports. We declare an import object `i` containing
one function `f` mentioned above. A string is encoded with a varint holding its
length followed by the string contents.

------------------------------------------------------------------------------
02 07  ; 7-byte import section.
01     ; One import.
01 69  ; The string "i".
01 66  ; The string "f".
00     ; Function.
00     ; Index of signature in section 1.
------------------------------------------------------------------------------

Section 3 declares signatures of the functions defined in section 10.

------------------------------------------------------------------------------
03 02  ; 2-byte function section.
01     ; One signature.
01     ; Index of signature in section 1.
------------------------------------------------------------------------------

Section 7 declares exports:

------------------------------------------------------------------------------
07 05  ; 5-byte export section.
01     ; One signature.
01 65  ; The string "e".
00     ; Function.
01     ; Index of signature in section 1.
------------------------------------------------------------------------------

Section 10 holds the code. Here, we define the body of our exported function,
which calls the imported function with the constant 42 (0x2a).

------------------------------------------------------------------------------
0a 08  ; 8-byte code section.
01     ; One function body.
06     ; Length of function body.
00     ; No local variables.
41 2a  ; Encoding of "i32.const 42".
10 00  ; Call function 0.
0b     ; End of code.
------------------------------------------------------------------------------

We can put this altogether in an HTML snippet that uses WebAssembly to show
"42" in an alert:

[source,html]
------------------------------------------------------------------------------
<script type="text/javascript">
WebAssembly.instantiate(new Uint8Array([
  0,97,115,109,1,0,0,0,1,8,2,96,1,127,0,96,0,0,2,7,1,1,105,1,
  102,0,0,3,2,1,1,7,5,1,1,101,0,1,10,8,1,6,0,65,42,16,0,11]),
  {i:{f:x => alert(x)}}).then(x => x.instance.exports.e());
</script>
------------------------------------------------------------------------------

== Parser ==

We modify link:../haskell/parse.html[a grammar from another demo] so it
accepts Haskell-style expressions: application associates to the left
and functions are curried. For example, `min 4 3`, the minimum of 4 and 3,
is interpreted as `(min 4) 3`, and `1 + 2` is really `((+) 1) 2`.

------------------------------------------------------------------------------
var    ::= ('a'|..|'z'|'A'|..|'Z')+
num    ::= ('0'|..|'9'|'.')+
factor ::= ['-'] ( var | num | '(' expr ')' )+
term   ::= factor ( ('*'|'/') factor )*
expr   ::= term ( ('+'|'-') term )*
line   ::= expr EOF
------------------------------------------------------------------------------

We parse an expression into simplified version of link:index.html[the data
structure we used to hold lambda calculus terms]. A leaf node either holds
a double constant, or a string representing a primitive function. All internal
nodes represent function application. Lambda abstraction is absent.

\begin{code}
{-# LANGUAGE CPP #-}
{-# LANGUAGE OverloadedStrings #-}
#ifdef __HASTE__
import Haste.DOM
import Haste.Events
import Haste.Foreign
import Data.List.Split
import Numeric
import System.IO.Unsafe
#else
import Data.ReinterpretCast
#endif
import Data.Char
import Text.ParserCombinators.Parsec

data Expr = D Double | V String | App Expr Expr

line :: Parser Expr
line = spaces >> expr >>= (eof >>) . pure where
  eat :: Parser a -> Parser a
  eat p = p >>= (spaces >>) . pure
  var  = eat $ V <$> many1 letter
  num  = eat $ D . read <$> many1 (digit <|> char '.')
  tok = eat . string
  una  = option id (tok "-" >> pure (App (V "neg")))
  fac  = una <*> (foldl1 App <$> many1
    (var <|> num <|> between (tok "(") (tok ")") expr))
  term = fac  `chainl1` ((tok "*" >> bin "*") <|> (tok "/" >> bin "/"))
  expr = term `chainl1` ((tok "+" >> bin "+") <|> (tok "-" >> bin "-"))
  bin s = pure $ \a b -> App (App (V s) a) b
\end{code}

== Compiler ==

Our compiler generates the above example with two changes:

  1. Our import function expects a double-precision floating point input
  (`f64`) instead of a 32-bit integer.

  2. Our function body consists of a translation of the given expression
  instead of the constant 42.

Rather than simply dump a wall of bytes, we write some helper routines to
output them so the structure of a wasm binary is apparent.

\begin{code}
toAsm e = concat [
  [0, 0x61, 0x73, 0x6d, 1, 0, 0, 0],  -- Magic string, version.
  -- Type section.
  sect 1 [encSig ["f64"] [], encSig [] []],
  -- Import section.
  -- [0, 0] = external_kind Function, index 0.
  sect 2 [encStr "i" ++ encStr "f" ++ [0, 0]],
  -- Function section.
  -- [1] = Type index.
  sect 3 [[1]],
  -- Export section.
  -- [0, 1] = external_kind Function, index 1.
  sect 7 [encStr "e" ++ [0, 1]],
  -- Code section.
  -- 0 = local variable count.
  -- [0x10, 0, 0xb] = call function 0, end of code.
  sect 10 [lenc $ 0 : compile [] e ++ [0x10, 0, 0xb]]]

lenc xs = length xs : xs

sect t xs = t : lenc (length xs : concat xs)

encStr s = lenc $ ord <$> s

encType "i32" = 0x7f
encType "f64" = 0x7c

encSig ins outs = 0x60  -- Function type.
  : lenc (encType <$> ins) ++ lenc (encType <$> outs)
\end{code}

No type checking is performed, though this is easy to add. Thus the only
non-trivial task is compiling the expression to assembly.

Wasm is stack-based, so for each operator:

 1. We recursively compile the subtrees for each of its operands (from left to
 right), so when executed, the operaands will be on top of the stack.

 2. We output the opcode corresponding to the operator.

We place the arity and opcodes for all the `f64` operators in an associative
list, which suffices for our simple demo.

In wasm, the 8 bytes of a double are encoded in little-endian.
Normally, we can take care of this in Haskell with, say,
https://hackage.haskell.org/package/reinterpret-cast/docs/Data-ReinterpretCast.html[the `Data.ReinterpretCast` package] but Haste lacks support for the
relevant Haskell primitives. For the Haste version,
we use
http://stackoverflow.com/questions/24564460/how-to-apply-bitwise-operations-to-the-actual-ieee-754-representation-of-js-numb[JavaScript
to get at the bytes representing a double]:

[source,html]
------------------------------------------------------------------------------
<script type="text/javascript">
function fromDouble(d) {return new Uint8Array(new Float64Array([d]).buffer);}
</script>
------------------------------------------------------------------------------

We wrap it in `unsafePerformIO` so we can use it as a pure function.

\begin{code}
f64una = zip (words "abs neg ceil floor trunc nearest sqrt") [0x99..]
f64bin = zip (words "+ - * / min max copysign") [0xa0..]

ops =
  ((\(x, b) -> (x, (1, b))) <$> f64una) ++
  ((\(x, b) -> (x, (2, b))) <$> f64bin)

compile m e = case e of
  App a b -> compile (b:m) a
#ifdef __HASTE__
  D d -> 0x44 : map fromIntegral (doubleToBytes d) where
    doubleToBytes :: Double -> [Int]
    doubleToBytes = unsafePerformIO . ffi "fromDouble"
#else
  D d -> 0x44 : map fromIntegral
    [doubleToWord d `div` (256^i) `mod` 256 | i <- [0..7]]
#endif
  V s -> case lookup s ops of
    Nothing -> error $ "bad op: " ++ s
    Just (a, b) -> concatMap (compile m) (take a m) ++ [b]
\end{code}

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
<script type="text/javascript">
function fromDouble(d) {return new Uint8Array(new Float64Array([d]).buffer);}
</script>
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

== User Interface ==

For the web version, we compile a given expression, dump the assembly in
a `textarea` element, and run the assembly which calls a function that shows
the result on the page:

[source,html]
------------------------------------------------------------------------------
<script type="text/javascript">
function runWasmInts(a) {
  WebAssembly.instantiate(new Uint8Array(a),
    {i:{f:x => Haste.setResult(x)}}).then(x => x.instance.exports.e());
}
</script>
------------------------------------------------------------------------------

For the command-line version of the program, we compile a predefined program.

\begin{code}
#ifdef __HASTE__
dump s = unlines $ unwords . map xxShow <$> chunksOf 8 s where
  xxShow c = reverse $ take 2 $ reverse $ '0' : showHex c ""

main = withElems ["input", "output", "asm", "evalB"] $
    \[iEl, oEl, aEl, evalB] -> do

  let
    setResult :: Double -> IO ()
    setResult d = setProp oEl "innerHTML" $ " = " ++ show d

  export "setResult" setResult
  evalB `onEvent` Click $ const $ do
    setProp oEl "innerHTML" ""
    ee <- parse line "" <$> getProp iEl "value"
    case ee of
      Left m -> do
        setProp aEl "value" ""
        setProp oEl "value" $ "parse error: " ++ show m
      Right e -> do
        let asm = toAsm e
        setProp aEl "value" $ dump asm
        ffi "runWasmInts" asm :: IO ()
#else
main = print $ toAsm e where Right e = parse line "" "min (sqrt 8) 2"
#endif
\end{code}
