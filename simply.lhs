= Solving the halting problem =

Soon after teaching Turing machines, educators often explain why the halting
problem is undecidable. But then they seem to leave the story unfinished.
Have we just learned we can never trust software? How can we rely on a program
to control spacecraft or medical equipment if it can unpredictably loop
forever?

One might claim extensive testing is the answer: we check a variety of cases
work as intended, then hope for the best. But though helpful, testing alone is
rarely satisfactory. An untested case may occur naturally and cause our code to
behave badly. Worse still, a malicious user could scour the untested cases to
find ways to deliberately sabotage our program.

The only real fix is to rein in those unruly Turing machines. By constraining
what can appear in our code, it might behave better. We might try banning GOTO
statements for example.

But how do we know if our restrictions are effective? Also, halting is but one
concern: even if we're sure our program halts, it should do the right thing.
Can we modify our languages so that we can prove program correctness?

It turns out we can only progress by replacing Turing machines with an
equally powerful model of computation link:.[lambda calculus].
Only then can we introduce types, and draw on logic and category theory to
prove results.

[pass]
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
<script src="simply.js"></script>
<p><button id="evalB">Run</button>
<button id="resetB">Reset</button>
</p>
<p><textarea style="border: solid 2px; border-color: #999999" id="input" rows="10" cols="80">
</textarea></p>
<p><textarea id="output" rows="8" cols="80" readonly></textarea></p>
<textarea id="resetP" hidden>
-- I once sat a multiple-choice exam where I lost more points for an incorrect
-- answer than I gained for a correct answer.
f = \x:I y:I->I b:B . if b then add x else y
goodPoints = 2
badFun = add (negate 3)
score = f goodPoints badFun
-- Compute the score after 2 right answers, a wrong answer, and a right answer.
score True (score False (score True (score True 0)))
</textarea>
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Unfortunately, some restrictions appear to have been invented without paying
any heed to theory. Could this be caused by the relative obscurity of lambda
calculus?

== Simply typed lambda calculus ==

We can easily modify lambda calculus so that all programs halt while retaining
a little power.  We'll walk through the solution that was first discovered, the
aptly named https://en.wikipedia.org/wiki/Simply_typed_lambda_calculus[simply
typed lambda calculus].

We start with 'base types', say `Int` and `Bool`, from which we
build other types with the `(->)` 'type constructor', such as:

------------------------------------------------------------------------------
Int -> Int -> Bool
------------------------------------------------------------------------------

Conventionally, `(->)` is right associative, so this means `Int -> (Int ->
Bool)`, namely, a function that takes an integer, and returns a function
mapping an integer to a boolean.

We can view this as a function that takes two `Int` parameters and returns
a single `Bool`. For example, the less-than function might have this type.
Representing functions that take multiple arguments as a series of functions
that each take one argument is known as 'currying'.

We populate the base types with 'constants', such as `0`, `1`, ... for `Int`,
and `True` and `False` for `Bool`.

This seems quotidian so far. Typical high-level languages do this sort
of thing. The fun part is seeing how easily it can be tacked on to lambda
calculus. There are only two changes:

  1. We add a new kind of leaf node, which holds a constant.
  2. The left child of a lambda abstraction (a variable) must be accompanied by
  a type.

We only need to add a few lines to our lambda calculus example to add simple
types. Let's get started:

\begin{code}
{-# LANGUAGE CPP #-}
#ifdef __HASTE__
import Haste.DOM
import Haste.Events
#else
import System.Console.Readline
#endif
import Control.Monad
import Data.Char
import Data.List
import Text.ParserCombinators.Parsec
\end{code}

First we need a new data structure to represent types. To avoid clashing
with predefined Haskell types, we use `B` for `Bool` and `I` for Int.

\begin{code}
data Type = I | B | Fun Type Type deriving Eq
\end{code}

By abuse of notation, `Var` will hold variables and constants: when the string
it holds is `True`, `False`, or a string representation of an integer, then it
counts as a constant. Otherwise it is a variable.

We add a type to the left child of a lambda abstraction.

We also add an `If` node, for if-then-else expressions. This has nothing to
do with simply typed lambda calculus, but we introduce it because, firstly,
we want to demonstrate how to extend lambda calculus, and secondly, simply
typed lambda calculus turns out to be a bit weak, and can use all the help
it can get!

\begin{code}
data Term = Var String | App Term Term | Lam (String, Type) Term
  | If Term Term Term

instance Show Type where
  show I = "I"
  show B = "B"
  show (Fun t u) = showL t ++ " -> " ++ show u where
    showL (Fun _ _) = "(" ++ show t ++ ")"
    showL _         = show t

instance Show Term where
  show (Lam (x, t) y)    = "\0955" ++ x ++ ":" ++ show t ++ showB y where
    showB (Lam (x, t) y) = " " ++ x ++ ":" ++ show t ++ showB y
    showB expr           = "." ++ show expr
  show (Var s)    = s
  show (App x y)  = showL x ++ showR y where
    showL (Lam _ _) = "(" ++ show x ++ ")"
    showL _         = show x
    showR (Var s)   = ' ':s
    showR _         = "(" ++ show y ++ ")"
  show (If x y z) = "if " ++ show x ++ " then " ++ show y ++ " else " ++ show z
\end{code}

== Parsing ==

Because we must attach type signatures to variable bindings, apart from adding
a parser for types, we also change our lambda calculus parser so that `(->)` is
strictly a type constructor, and `(.)` is strictly for lambda abstractions.
Haskell gets away with using `(->)` for both cases because its grammar is
different. (For example, we can declare the type of a Haskell function
in one line, and define it in another.)

The strings ``if'', ``then'', and ``else'' are reserved keywords and hence
invalid variable names.

\begin{code}
data LambdaLine = Empty | Let String Term | Run Term

line :: Parser LambdaLine
line = (((eof >>) . pure) =<<) . (ws >>) $ option Empty $ do
  t <- term
  option (Run t) $ str "=" >> Let (getV t) <$> term where
  getV (Var s) = s
  term = ifthenelse <|> lam <|> app
  ifthenelse = do
    str "if"
    cond <- term
    str "then"
    bfalse <- term
    str "else"
    btrue <- term
    pure $ If cond bfalse btrue
  lam = flip (foldr Lam) <$> between lam0 lam1 (many1 vt) <*> term where
    lam0 = str "\\" <|> str "\0955"
    lam1 = str "."
    vt   = do
      s <- v
      str ":"
      t <- typ
      pure $ (s, t)
  typ = ((str "B" >> pure B) <|> (str "I" >> pure I)
    <|> between (str "(") (str ")") typ)
      `chainr1` (str "->" >> pure Fun)
  app = foldl1' App <$> many1 ((Var <$> v) <|> between (str "(") (str ")") term)
  v   = try $ do
    s <- many1 alphaNum
    when (s `elem` ["if", "then", "else"]) $ fail "unexpected keyword"
    ws
    pure s
  str = (>> ws) . string
  ws = spaces >> optional (try $ string "--" >> many anyChar)
\end{code}

== Typing ==

In a closed lambda term, every leaf node is typed because it's either a
constant, or its type is given at its binding. Type checking works in
the obvious manner: for example, we can only apply a function of type
`Int -> Int -> Bool` to an `Int`, and we can only apply the resulting function
to an `Int`, and the result will be a `Bool`.

We predefine a few functions: `negate`, `add`, and `not`, whose types are
hard-coded here.

Traditionally, an uppercase gamma denotes a set of variables and their types,
which is called a 'typing context' or 'typing environment', hence our use of
`gamma` for the first argument in our `typeOf` function:

\begin{code}
typeOf gamma t = case t of
  Var "negate" -> Just (Fun I I)
  Var "add" -> Just (Fun I (Fun I I))
  Var "not" -> Just (Fun B B)
  Var "False" -> Just B
  Var "True"  -> Just B
  Var s | [(_, [])] <- (reads s :: [(Integer, String)]) -> Just I
        | otherwise                                     -> lookup s gamma
  App x y -> do
    tx <- rec x
    ty <- rec y
    case tx of
      Fun ty' tz | ty == ty' -> pure tz
      _                      -> Nothing
  Lam (x, t) y -> do
    u <- typeOf ((x, t):gamma) y
    pure $ Fun t u
  If x y z -> if rec x /= Just B then Nothing else do
    ty <- rec y
    tz <- rec z
    if ty == tz then pure ty else Nothing
  where rec = typeOf gamma
\end{code}

== Evaluation ==

Evaluation works the same as it does for untyped lambda calculus. In fact,
we could perform 'type erasure' and drop the type of every bound variable
before evaluation: types are only needed during type checking. However,
keeping types around can be useful for sanity checks. (The GHC compiler has
a typed intermediate language for this reason.)

Otherwise, apart from handling predefined functions, our `eval` function
is the same as our corresponding function of untyped lambda calculus.

We assume the input is well-typed and hence closed. This implies normalization
is much simpler than with untyped lambda calculus: we have no need for the
`norm` function.

\begin{code}
eval env (If x y z) = eval env $ case eval env x of
  Var "True"  -> y
  Var "False" -> z

eval env (App m a) = let m' = eval env m in case m' of
  Lam (v, _) f -> let
    beta (Var s) | s == v         = a
                 | otherwise      = Var s
    beta (Lam (s, t) m)
                 | s == v         = Lam (s, t) m
                 | s `elem` fvs   = let s1 = newName s fvs in
                   Lam (s1, t) $ beta $ rename s s1 m
                 | otherwise      = Lam (s, t) (beta m)
    beta (App m n)                = App (beta m) (beta n)
    beta (If x y z)               = If (beta x) (beta y) (beta z)
    fvs = fv env [] a
    in eval env $ beta f
  Var "not" -> case eval env a of
    Var "True"  -> Var "False"
    Var "False" -> Var "True"
  Var "negate" -> case eval env a of
    Var x  -> Var (show $ negate $ (read x :: Integer))
  App (Var "add") b -> Var $ show (m + n) where
    Var mstr = eval env a
    Var nstr = eval env b
    m = read mstr :: Integer
    n = read nstr :: Integer
  _ -> App m' a 

eval env term@(Var v) | Just x  <- lookup v env = eval env x
eval _   term                                   = term

fv env vs (Var s) | s `elem` vs            = []
                  | Just x <- lookup s env = fv env (s:vs) x
                  | otherwise              = [s]
fv env vs (App x y)                        = fv env vs x `union` fv env vs y
fv env vs (Lam (s, _) f)                   = fv env (s:vs) f

newName x ys = head $ filter (`notElem` ys) $ (s ++) . show <$> [1..] where
  s = dropWhileEnd isDigit x

rename x x1 term = case term of
  Var s | s == x    -> Var x1
        | otherwise -> term
  Lam (s, t) b
        | s == x    -> term
        | otherwise -> Lam (s, t) (rec b)
  App a b           -> App (rec a) (rec b)
  where rec = rename x x1
\end{code}

== User interface ==

The outer layer of our program differs from the one we wrote for untyped
lambda calculus in that we check a term is well-typed before adding it to our
list of let definitions or evaluating it. We print the type of terms to show
off our type checking.

For each let definition, we record the definition as well as its type.
Unlike our untyped lambda calculus interpreter, recursion is forbidden,
because we require the body of a let definition to be well-typed, and we
only prepopulate `gamma` with the types of previous let definitions.

\begin{code}
#ifdef __HASTE__
main = withElems ["input", "output", "evalB", "resetB", "resetP"] $
  \[iEl, oEl, evalB, resetB, resetP] -> do
  let
    reset = getProp resetP "value" >>= setProp iEl "value" >> setProp oEl "value" ""
  reset
  resetB `onEvent` Click $ const reset
  evalB `onEvent` Click $ const $ do
    let
      run (out, env) (Left err) =
        (out ++ "parse error: " ++ show err ++ "\n", env)
      run (out, env@(gamma, lets)) (Right m) = case m of
        Empty      -> (out, env)
        Run term   -> case typeOf gamma term of
          Nothing -> (out ++ "type error: " ++ show term ++ "\n", env)
          Just t  -> (out ++ show (eval lets term) ++ "\n", env)
        Let s term -> case typeOf gamma term of
          Nothing -> (out ++ "type error: " ++ show term ++ "\n", env)
          Just t  -> (out ++ "[" ++ s ++ ":" ++ show t ++ "]\n",
            ((s, t):gamma, (s, term):lets))
    es <- map (parse line "") . lines <$> getProp iEl "value"
    setProp oEl "value" $ fst $ foldl' run ("", ([], [])) es
#else
repl env@(gamma, lets) = do
  let redo = repl env
  ms <- readline "> "
  case ms of
    Nothing -> putStrLn ""
    Just s  -> do
      addHistory s
      case parse line "" s of
        Left err  -> do
          putStrLn $ "parse error: " ++ show err
          redo
        Right Empty -> redo
        Right (Run term) -> do
          case typeOf gamma term of
            Nothing -> putStrLn $ "type error"
            Just t -> do
              putStrLn $ "[type = " ++ show t ++ "]"
              print $ eval lets term
          redo
        Right (Let s term) -> case typeOf gamma term of
          Nothing -> putStrLn ("type error") >> redo
          Just t -> do
            putStrLn $ "[type = " ++ show t ++ "]"
            repl ((s, t):gamma, (s, term):lets)

main = repl ([], [])
#endif
\end{code}

With induction, we can show type checking is efficient, and if a closed lambda
term is well-typed, then it's guaranteed to have a normal form. (In particular,
the Y combinator and omega combinator cannot be expressed in this system.)
Moreover, any evaluation strategy will lead to the normal form, that is, simply
typed lambda calculus is 'strongly normalizing'.

In other words, programs always halt. Try doing this with Turing machines!

== Too simple? ==

The above language is restrictive. For example, loops are impossible!

Limited power is fine for certain applications. In fact, it turns out some
forms of typed lambda calculus wind up embedding simply typed lambda calculus
in the type system itself. But for general-purpose programming, we need more.

There are two ways to fix this:

  1. We boldly add features like unrestricted recursion. We lose our guarantee
  that all programs halt, but at least most of our language is trustworthy.

  2. We cautiously enrich our type system, at each step checking that
  well-typed programs normalize. With a sufficiently advanced type system,
  we gain some forms of recursion.
