= Lambda Calculus =

Back when I took computer science courses, they taught us Turing machines.
Despite being purely theoretical, Turing machines are important:

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
<p><button id="evalB">Run</button>
<button id="factB">Factorial</button></p>
<p><textarea style="border: solid 2px; border-color: #999999" id="input" rows="10" cols="80">2 = \f x -> f (f x)
3 = \f x -> f (f (f x))
exp = \m n -> n m
exp 2 3
</textarea></p>
<p><textarea id="output" rows="3" cols="80" readonly></textarea></p>
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

== Why Lambda Calculus? ==

Firstly, there's the historical significance. Alonzo Church was Alan Turing's
doctoral advisor, and his lambda calculus predates Turing machines. But more
importantly, working through the theory from its original viewpoint exposes
us to different ways of thinking. Aside from a healthy mental workout, we find
the lambda calculus approach is sometimes superior.

For example, soon after teaching Turing machines, educators often show why the
halting problem is undecidable. But my textbooks seemed to leave the story
unfinished. Vexing questions spring to mind. Have we just learned we can
never trust software? How can we rely on a program to control spacecraft or
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
is https://en.wikipedia.org/wiki/Conway's_Game_of_Life[Conway's Game of
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
Haskell a little further and use `->` instead of periods, and support line
comments.

Any alphanumeric string is a valid variable name.

Typing a long term is tedious, so we support a sort of 'let' statement. The line

------------------------------------------------------------------------------
true = \x y -> x
------------------------------------------------------------------------------

means that for all following terms, the variable `true` is no longer a
variable, but shorthand for the term on the right side, namely `\x y -> x`.
There is one exception: if the variable `true` is the left child of a lambda
abstraction, then it remains unexpanded and counts as a variable; ideally we'd
pick a different name to avoid confusion.

\begin{code}
line :: Parser (Maybe (String, Term))
line = (ws >>) $ optionMaybe $ do
  t <- term
  r <- option ("", t) $ str "=" >> (,) (getV t) <$> term
  eof
  pure r where
  getV (Var s) = s
  term = lam <|> app
  lam = flip (foldr Lam) <$> between (str "\\") (str "->") (many1 v) <*> term
  app = foldl1' App <$> many1 ((Var <$> v) <|> between (str "(") (str ")") term)
  v   = many1 alphaNum >>= (ws >>) . pure
  str = (>> ws) . string
  ws = spaces >> optional (try $ string "--" >> many anyChar)
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
(beta reductions) are possible, leaving the term in 'head normal form'.
We recursively call `eval` on child nodes to reduce other function applications
throughout the tree, resulting in the 'normal form' of the lambda term. The
normal form is unique in some sense.

If we start from an expression with no free variables, that is, a 'closed
lambda expression' or 'combinator', then after our code finishes, there
should be no more `App` nodes.

There's no guarantee that our recursion terminates. For example, it is
impossible to reduce all the `App` nodes of:

------------------------------------------------------------------------------
omega = (\x -> x x)(\x -> x x)
------------------------------------------------------------------------------

In such cases, we say the lambda term has no normal form. We could limit the
number of reductions to prevent our code looping forever; we leave this as an
exercise for the reader.

Viewing lambda terms as a binary tree again, we see `eval` is an in-order
tree algorithm. This is called a 'normal-order evaluation strategy'.
We could have also tried post-order traversal, that is, we evaluate the child
nodes before the parent. This is called 'applicative order', and unlike normal
order, it fails to normalize some terms that in fact possess a normal form.

\begin{code}
norm env term = case eval env term of
  App x y -> App x $ norm env y
  Lam v f -> Lam v $ norm env f
  Var x   -> Var x
\end{code}

Lastly, the user interface:

\begin{code}
#ifdef __HASTE__
main = withElems ["input", "output", "evalB", "factB", "factP"] $
  \[iEl, oEl, evalB, factB, factP] -> do
  factB `onEvent` Click $ const $ do
    getProp factP "value" >>= setProp iEl "value"
    setProp oEl "value" ""
  evalB `onEvent` Click $ const $ do
    let
      run (out, env) (Left err) =
        (out ++ "parse error: " ++ show err ++ "\n", env)
      run (out, env) (Right m) = case m of
        Nothing         -> (out, env)
        Just ("", term) -> (out ++ show (norm env term) ++ "\n", env)
        Just (s , term) -> (out, (s, term):env)
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
        Right Nothing -> repl env
        Right (Just ("", term)) -> do
          print $ norm env term
          repl env
        Right (Just (s,  term)) -> repl ((s, term):env)

main = repl []
#endif
\end{code}

[pass]
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
<textarea id="factP" hidden>true = \x y -> x
false = \x y -> y
0 = \f x -> x
1 = \f x -> f x
succ = \n f x -> f(n f x)
pred = \n f x -> n(\g h -> h (g f)) (\u -> x) (\u ->u)
mul = \m n f -> m(n f)
is0 = \n -> n (\x -> false) true
Y = \f -> (\x -> x x)(\x -> f(x x))
fact = Y(\f n -> (is0 n) 1 (mul n (f (pred n))))
fact (succ (succ (succ 1)))
</textarea>
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

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
machines. We seem to endlessly substitute functions in other functions; they
never seem to ``bottom out''. Even 1 + 1 seems hard to represent!

The trick is to use functions to represent data. This is less obvious than
encoding Turing machines on a tape, but well worth learning. The original and
most famous scheme is known as 'Church encoding'.
See http://www.cs.yale.edu/homes/hudak/CS201S08/lambda.pdf['A Brief and
Informal Introduction to the Lambda Calculus'] by Paul Hudak,
and https://en.wikipedia.org/wiki/Church_encoding[Wikipedia's entry on Church
encoding] for details. We'll only summarize briefly

Booleans look cute in the Church encoding:

------------------------------------------------------------------------------
true = \x y -> x
false = \x y -> y
and = \p q -> p q p
or = \p q -> p p q
if = \p x y -> p x y
ifAlt = \p -> p  -- So "if" can be omitted in programs!
not = \p -> p false true
notAlt = \p x y -> p y x
------------------------------------------------------------------------------

Integers are encoded in a unary manner. The predecessor function is far slower
than the successor function, as it constructs the answer by starting from 0
and epeatedly computing the successor. There is no quick way to ``strip off''
one layer of a function application.

------------------------------------------------------------------------------
0 = \f x -> x
1 = \f x -> f x
2 = \f x -> f (f x)
-- ...and so on.
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

Also, we can pair up any two terms:

------------------------------------------------------------------------------
pair = \x y z -> z x y
fst = \p -> p true
snd = \p -> p false
------------------------------------------------------------------------------

From such tuples, we can construct lists, trees, and so on.

Incidentally, we'd have a faster predecessor function if we used
https://ifl2014.github.io/submissions/ifl2014_submission_13.pdf[the Scott encoding]:

------------------------------------------------------------------------------
0 = \f x -> x
succ = \n f x -> f n
pred = \n -> n (\x -> x) 0
is0 = \n -> n (\x -> false) true
------------------------------------------------------------------------------

== Recursion ==

Because our interpreter cheats and only looks up a let definition at the last
minute, we can recursively compute factorials with:

------------------------------------------------------------------------------
factrec = \n -> if (is0 n) 1 (mul n (factrec (pred n)))
------------------------------------------------------------------------------

But we stress this is not a lambda calculus term. If we tried to expand the let
definitions, we'd be forever replacing `factrec` with an expression containing
a `factrec`. We can never eliminate all the function names and reach a valid
lambda calculus term.

Instead, we need something like the
https://en.wikipedia.org/wiki/Fixed-point_combinator['Y combinator']. The inner
workings are described in many other places, so we'll content ourselves
with listing their definitions, and observing they are indeed lambda calculus
terms.

------------------------------------------------------------------------------
Y = \f -> (\x ->f(x x))(\x -> f(x x))
fact = Y(\f n -> if (is0 n) 1 (mul n (f (pred n))))
------------------------------------------------------------------------------

Thus we can simulate any Turing machine with a lambda calculus term: we could
concoct a data structure to represent a tape, which we'd feed into a recursive
function that carries out the state transitions.

== Practical lambda calculus ==

The above factorial functions are shorter than equivalent code in many
high-level languages. Indeed, unlike Turing machines, we can turn lambda
calculus into a practical programming langauge with just a few tweaks.

We've already seen one such tweak: we can allow recursion by expanding a
let definition on demand.

As for inefficient encoding schemes: we can define primitive types and data
structures. In fact, link:lisp.html[by adding bits and pieces to lambda
calculus, we wind up with a practical programming language: Lisp]. Haskell is
similar.

However, there's a much tougher problem: real programs have side effects. After
all, why bother computing a number if there's no way to print it?

The most obvious solution is to allow 'impure functions', that is, functions
that may print to the screen or have some other side effect when evaluated.
This requires us to carefully specify exactly when a function is evaluated.
Lisp does this by stipulating applicative order, so we can reason about the
ordering of side effects, and provides a macro feature to override applicative
order for special cases. Unfortunately, we lose nice features of the theory:
notably, some programs that would halt with a normal-order evaluation strategy
will loop forever.

It turns out there are other solutions that keep functions pure and hence
support normal-order evaluation, one of which is used by Haskell.

== Taming Turing Machines ==

Type systems are where lambda calculus really outshines Turing machines.

In the aptly named
https://en.wikipedia.org/wiki/Simply_typed_lambda_calculus[simply typed lambda
calculus], we start with 'base types', say `Int` and `Bool`, from which we
build other types with the `(->)` 'type constructor', such as:

------------------------------------------------------------------------------
Int -> Int -> Bool
------------------------------------------------------------------------------

Conventionally, `(->)` is right associative, so this means `Int -> (Int ->
Bool)`, which we interpret as a function that takes an integer, and returns
a function mapping an integer to a boolean. The less-than function would have
this type.

We populate the base types with 'constants', such as `0`, `1`, ... for `Int`,
and `True` and `False` for `Bool`.

So far, this all seems boring, and resembles what a typical high-level language
defines. The fun part is seeing how easily it can be tacked on to lambda
calculus. There are only two changes:

  1. We add a new kind of leaf node, which holds a constant.
  2. The left child of a lambda abstraction (a variable) must be accompanied by
  a type.

We might modify our data types as follows:

------------------------------------------------------------------------------
data Type = Int | Bool | Fun Type Type
data Term = Con String | Var String | App Term Term | Lam String Type Term
------------------------------------------------------------------------------

though in reality we'd rename to avoid clashing with predefined types.

Then in a closed lambda term, every leaf node is typed because it's either a
constant, or its type is given at its binding. Type checking works in
the obvious manner: for example, we can only apply a function of type
`Int -> Int -> Bool` to an `Int`, and we can only apply the resulting function
to an `Int`, and the result will be a `Bool`.

It can be shown that type checking is efficient, and if a closed lambda term
is correctly typed, then it's guaranteed to have a normal form. (In particular,
the Y combinator and omega combinator cannot be expressed in this system.)
Moreover, any evaluation strategy will lead to the normal form, that is, simply
typed lambda calculus is 'strongly normalizing'.

In other words, programs always halt. If we're allowing recursion, then this is
no longer true, but at least it narrows down the suspect parts of our program;
furthermore, by restricting recursion in certain ways, we can regain the
assurance that our programs will halt.

Try doing this with Turing machines!

We've only scratched the surface. Lambda calculus is but one interesting node
in a giant tree that includes: self-interpreters; combinatory logic; type
inference; more expressive type systems; proving program correctness.
