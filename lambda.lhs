= Lambda Calculus =

My computer science courses included Turing machines. Despite being purely
theoretical, Turing machines are important:

 - A state machine reading to and writing from cells on an infinite tape is a
 useful abstraction of a CPU reading from and writing to RAM.
 - Even at higher levels, popular programming languages closely adhere to the
 same model: a program writes data to memory, then later reads it to decide a
 future course of action.
 - We immediately see how to measure algorithmic complexity.
 - Encoding a Turing machine on a tape is straightforward, and gently guides
 us to see the equivalence of code and data, with all its deep implications.

All the same, I wish they had also taught us an alternative model of
computation known as https://en.wikipedia.org/wiki/Lambda_calculus['lambda calculus']:

[pass]
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
<script src="lambda.js"></script>
<p><button id="evalB">Run</button></p>
<p><textarea style="border: solid 2px; border-color: #999999" id="input" rows="10" cols="80">2 = \f x -> f (f x)
3 = \f x -> f (f (f x))
exp = \m n -> n m
exp 2 3
</textarea></p>
<p><textarea id="output" rows="3" cols="80" readonly></textarea></p>
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Firstly, there's the historical significance. Alonzo Church was Alan Turing's
doctoral advisor, and his lambda calculus predates Turing machines. But more
importantly, working through the theory from its original viewpoint exposes
us to different ways of thinking. Aside from a healthy mental workout, we find
the lambda calculus approach is sometimes superior.

For example, soon after teaching Turing machines, educators often show why the
halting problem is undecidable. But my textbooks seemed to leave the story
unfinished. Vexing questions spring to mind. Have we just learned that software
can never be trusted? How can we rely on a program to control spacecraft or
medical equipment if it can unpredictably loop forever?

One might claim extensive testing is the answer: we check a bunch of common
cases and edge cases work as intended, then hope for the best. But though
helpful, testing alone is rarely satisfactory. An untested case may occur
naturally and cause our code to behave badly. Worse still, a malicious user
could scour the untested cases to find ways to deliberately sabotage our
program.

The only real fix is to rein in those unruly Turing machines. By constraining
what can appear in our code, we can prove it behaves. We could ban GOTO
statements, or try something more heavy-handed like a type system.

Unfortunately, programmers appear to have invented some restrictions without
paying any heed to theory. Could this be caused by education systems
ignoring lambda calculus? Restricting lambda calculus is easier than
restricting Turing machines.

== Beta reduction ==

Lambda calculus starts off far simpler than Turing machines. In school, we're
accustomed to evaluating functions. In fact, one might argue they focus too
much on making students memorize and apply formulas: for instance,
$\sqrt{a^2 + b^2}$ for $a = 3$ and $b = 4$. In lambda calculus, this is called
'beta reduction', though instead of numbers like 3 and 4, we're plugging in
other formulas.

I was surprised this substitution process learned in childhood is all we need
for computing anything. A Turing machine has states, a tape of cells, and a
movable head that reads and writes; how can putting formulas into formulas be
equivalent? [In retrospect, maybe my surprise was unwarranted.
https://en.wikipedia.org/wiki/Tag_system[Tag systems] are Turing-complete, as
is is https://en.wikipedia.org/wiki/Conway's_Game_of_Life[Conway's Game of
Life].]

Our description is vague. The details will become clear as we build our
interpreter. Afterwards, we'll see how to compute anything with it.

As the imports suggest, we can build a command-line interpreter or a JavaScript
interpreter for this webpage from our source:

\begin{code}
{-# LANGUAGE CPP #-}
#ifdef __HASTE__
import Haste.DOM
import Haste.Events
#else
import System.Console.Readline
#endif
import Data.Char
import Data.List
import Text.ParserCombinators.Parsec
\end{code}

Lambda calculus terms can be viewed as a kind of full binary tree. A lambda
calculus term consists of:

  * 'Variables', which we can think of as leaf nodes holding strings.
  * 'Applications', which we can think of as internal nodes.
  * 'Lambda abstractions', which we can think of as a special kind of internal
  node whose left child must be a variable.

When printing terms, we'll use Unicode to show a lowercase lambda (&#0955;).
Conventionally:

  * Function application has higher precedence, associates to the left, and
    their child nodes are juxtaposed.
  * Lambda abstractions associate to the right, are prefixed with a lowercase
    lambda, and their child nodes are separated by periods.
  * With consecutive bindings (e.g. "λx0.λx1...λxn."), we omit all lambdas but
    the first, and omit all periods but the last (e.g. "λx0 x1 ... xn.").

For clarity, we enclose lambdas in parentheses if they are right child of an
application.

\begin{code}
data Term = Var String | App Term Term | Lam String Term

instance Show Term where
  show (Lam x y)  = "\0955" ++ x ++ showB y where
    showB (Lam x y) = " " ++ x ++ showB y
    showB expr      = "." ++ show expr
  show (Var s)    = s
  show (App x y)  = showL x ++ showR y where
    showL (Lam _ _ ) = "(" ++ show x ++ ")"
    showL _          = show x
    showR (Var s)    = ' ':s
    showR _          = "(" ++ show y ++ ")"
\end{code}

When reading terms, since typing Greek letters can be nontrivial, we
follow Haskell and interpret the backslash as lambda. We may as well follow
Haskell a little further and use `->` instead of periods.

Any alphanumeric string is a valid variable name.

Typing a long term is tedious, so we support a sort of let statement. The line

------------------------------------------------------------------------------
true = \x y -> x
------------------------------------------------------------------------------

means that for all following terms, the variable `true` is no longer a
variable, but shorthand for the term on the right side, namely `\x y -> x`.
There is one exception: if the variable `true` is the left child of a lambda
abstraction, then it is left unexpanded and counts as a variable, but ideally
we should pick a different name to avoid confusion.

\begin{code}
line :: Parser (String, Term)
line = do
  spaces
  t <- term
  r <- option ("", t) $ str "=" >> (,) (getV t) <$> term
  eof
  pure r where
  getV (Var s) = s
  term = lam <|> app
  lam = flip (foldr Lam) <$> between (str "\\") (str "->") (many1 v) <*> term
  app = foldl1' App <$> many1 ((Var <$> v) <|> between (str "(") (str ")") term)
  v   = many1 alphaNum >>= (spaces >>) . pure
  str = (>> spaces) . string
\end{code}

== Evaluation ==

If the root node is a free variable or a lambda, then there is nothing to do.
Otherwise, the root node is an App node, and we recursively evaluate the left
child.

The left child should evaluate to a lambda; if not, then we stop, as a free
variable got in the way somewhere.

Otherwise, we perform beta reduction. Thus we traverse the tree and replace
leaf nodes representing our variable with a certain subtree. However, there
is one potential complication: we must never change a 'free variable' into
a 'bound variable', which we accomplish by renaming variables. For example,
reducing `(\y -> \x -> y)x` to `\x -> x` is incorrect, so we rename the
first occurence of `x` to get `\x1 -> x`.

More precisely, a variable `x` is bound if appears in the right subtree of a
lambda node whose left child is also `x`, and free otherwise. If a substitution
would cause a free variable to become bound, then we rename all free occurrences
of that variable before proceeding. The new name must differ from all other
free variables, so we must find all free variables.

We store the let definitions in an associative list named `env`, and perform
lookups on demand to see if a given string is a variable or shorthand for
another term.

We're supposed to expand all such definitions before beginning evaluation, so
that we can be sure the input is a valid term, but this lazy approach enables
recursive let definitions. Thus our interpreter actually runs more than plain
lambda calculus, in the same way that Haskell allows unrestricted recursion
tacked on a typed variant of lambda calculus.

\begin{code}
eval env term@(App x a) | Lam v f <- eval env x   = let
  beta (Var s)   | s == v         = a
                 | otherwise      = Var s
  beta (Lam s y) | s == v         = Lam s y
                 | s `elem` frees = let s1 = newName s frees in
                    Lam s1 $ beta $ rename s s1 y
                 | otherwise      = Lam s (beta y)
  beta (App x y)                  = App (beta x) (beta y)
  frees = fv [] a
  fv vs (Var s) | s `elem` vs            = []
                | Just x <- lookup s env = fv (s:vs) x
                | otherwise              = [s]
  fv vs (App x y)                        = fv vs x `union` fv vs y
  fv vs (Lam s f)                        = fv (s:vs) f
  in eval env $ beta f
eval env term@(Var v)   | Just x  <- lookup v env = eval env x
eval _   term                                     = term
\end{code}

To pick a new name, we increment the number at the end of the name (or append
"1" if there is no number) until we've avoided clashing with an existing name:

\begin{code}
newName x ys = head $ filter (`notElem` ys) $ (s ++) . show <$> [1..] where
  s = dropWhileEnd isDigit x
\end{code}

Renaming a free variable is a tree traversal that skips lambda subtrees if
their left child matches the variable being renamed:
 
\begin{code}
rename x x1 term = case term of
  Var s   | s == x    -> Var x1
          | otherwise -> term
  Lam s b | s == x    -> term
          | otherwise -> Lam s (rec b)
  App a b             -> App (rec a) (rec b)
  where rec = rename x x1
\end{code}

Our `eval` function terminates once no more top-level function applications
are possible, leaving the term in 'weak head normal form'. To fully normalize
terms, we recursively call `eval` on child nodes. (This loops forever for
terms lacking a normal form.)

\begin{code}
norm env term = case eval env term of
  App x y -> App x $ norm env y
  Lam v f -> Lam v $ norm env f
  Var x   -> Var x
\end{code}

Lastly, the user interface:

\begin{code}
{-# LANGUAGE CPP #-}
#ifdef __HASTE__
main = withElems ["input", "output", "evalB"] $
  \[iEl, oEl, evalB] -> do
  evalB `onEvent` Click $ const $ do
    let
      run (out, env) term = case term of
        Left err         -> (out ++ "parse error: " ++ show err ++ "\n", env)
        Right ("", term) -> (out ++ show (norm env term) ++ "\n", env)
        Right (s , term) -> (out, (s, term):env)
    es <- map (parse line "") . lines <$> getProp iEl "value"
    setProp oEl "value" $ fst $ foldl' run ("", []) es
#else
repl env = do
  ms <- readline "> "
  case ms of
    Nothing -> putStrLn ""
    Just s  -> do
      addHistory s
      case parse line "" s of
        Left err  -> do
          putStrLn $ "parse error: " ++ show err
          repl env
        Right ("", term) -> do
          print $ norm env term
          repl env
        Right (s,  term) -> repl ((s, term):env)

main = repl []
#endif
\end{code}

== A Lesson Learned ==

Until I wrote an interpreter, my understanding of renaming was flawed. I knew
that we compute with 'closed terms', that is terms with no free variables, so I
had thought this meant I could skip implementing renaming.  No free variables
can become bound because they're all bound to begin with, right?

In an early version of this interpreter, I tried to normalize:

------------------------------------------------------------------------------
(\f x -> f x)(\f x -> f x)
------------------------------------------------------------------------------

My old program mistakenly returned:

------------------------------------------------------------------------------
\x x -> x x
------------------------------------------------------------------------------

It's probably obvious to others, but at last I realized the problem is
that beta reduction recurses on subtrees, thus in the right subtree of a lambda
abstraction, a variable may be free, even though it is bound when the entire
tree is considered. With renaming, my program gave the correct answer:

------------------------------------------------------------------------------
\x x1 -> x x1
------------------------------------------------------------------------------

== Church encoding ==

When starting out with lambda calculus, we soon miss the symbols of Turing
machines. We endlessly substitute functions in other functions; they never seem
to ``bottom out''. Even 1 + 1 seems hard to represent!

The trick is to use functions to represent data. This is less obvious than
encoding Turing machines on a tape, but well worth learning. The original and
most famous scheme is known as 'Church encoding'.

The following booleans, natural numbers, and functions that work on them are
from http://www.cs.yale.edu/homes/hudak/CS201S08/lambda.pdf['A Brief and
Informal Introduction to the Lambda Calculus'] by Paul Hudak,
and https://en.wikipedia.org/wiki/Church_encoding[Wikipedia's entry on Church
encoding]:

------------------------------------------------------------------------------
true = \x y -> x
false = \x y -> y
if = \p x y -> p x y
and = \p q -> p q p
0 = \f x -> x
1 = \f x -> f x
2 = \f x -> f (f x)
...
succ = \n f x -> f(n f x)
pred = \n f x -> n(\g h -> h (g f)) (\u -> x) (\u ->u)
add = \m n f x -> m f(n f x)
sub = \m n -> (n pred) m
mul = \m n f -> m(n f)
exp = \m n -> n m
is0 = \n -> n (\x -> false) true
le = \m n -> is0 (sub m n)
eq = \m n -> and (le m n) (le n m)
------------------------------------------------------------------------------

== Recursion ==

Because our interpreter cheats, we can recursively compute factorials with:

------------------------------------------------------------------------------
factrec = \n -> if (is0 n) 1 (mul n (factrec (pred n)))
------------------------------------------------------------------------------

However, in lambda calculus, recursion requires something like the
Y-combinator. The inner workings are described in many other references, so
we'll content ourselves with listing their definitions, and noting they
are valid lambda calculus terms.

------------------------------------------------------------------------------
Y = \f -> (\x ->f(x x))(\x -> f(x x))
fact = Y(\f n -> if (is0 n) 1 (mul n (f (pred n))))
------------------------------------------------------------------------------
