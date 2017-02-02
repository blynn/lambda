= Lambda Calculus =

They taught us Turing machines in my computer science classes.
Despite being purely theoretical, Turing machines are important:

 - A state machine reading to and writing from cells on an infinite tape is a
 useful abstraction of a CPU reading from and writing to RAM.
 - Many high-level programming languages adhere to the same model: code writes
   data to memory, then later reads it to decide a future course of action.
 - We immediately see how to measure algorithmic complexity.
 - Encoding a Turing machine on a tape is straightforward, and gently guides
 us to the equivalence of code and data, with all its deep implications.

All the same, I wish they had also taught us an alternative model of
computation known as https://en.wikipedia.org/wiki/Lambda_calculus['lambda calculus']:

[pass]
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
<script src="lambda.js"></script>
<p><button id="evalB">Run</button>
<button id="factB">Factorial</button>
<button id="surB">Surprise Me!</button></p>
<p><textarea style="border: solid 2px; border-color: #999999" id="input" rows="12" cols="80">2 = \f x -> f (f x)
3 = \f x -> f (f (f x))
exp = \m n -> n m
exp 2 3  -- Compute 2^3.
</textarea></p>
<p><textarea id="output" rows="3" cols="80" readonly></textarea></p>
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

== Why Lambda Calculus? ==

Lambda calculus is historically significant. Alonzo Church was Alan Turing's
doctoral advisor, and his lambda calculus predates Turing machines.

But more importantly, working through the theory from its original viewpoint
exposes us to different ways of thinking. Aside from a healthy mental workout,
we find the lambda calculus approach is sometimes superior.

For example, soon after teaching Turing machines, educators often show why the
halting problem is undecidable. But my textbooks seemed to leave the story
unfinished. Vexing questions spring to mind. Have we just learned we can
never trust software? How can we rely on a program to control spacecraft or
medical equipment if it can unpredictably loop forever?

One might claim extensive testing is the answer: we check a variety of cases
work as intended, then hope for the best. But though helpful, testing alone is
rarely satisfactory. An untested case may occur naturally and cause our code to
behave badly. Worse still, a malicious user could scour the untested cases to
find ways to deliberately sabotage our program.

The only real fix is to rein in those unruly Turing machines. By constraining
what can appear in our code, we can prove it behaves. We could ban GOTO
statements, or try something more heavy-handed like a type system.

Unfortunately, programmers appear to have invented some restrictions without
paying any heed to theory. Could this be caused by education systems
ignoring lambda calculus? Restricting lambda calculus is easier than
restricting Turing machines.

== Beta reduction ==

Unlike Turing machines, everyone already knows the basics of lambda calculus.
In school, we're accustomed to evaluating functions. In fact, one might argue
they focus too much on making students memorize and apply formulas such as
$\sqrt{a^2 + b^2}$ for $a = 3$ and $b = 4$.

In lambda calculus, this is called 'beta reduction', and we'd write this
example as:

\[ (\lambda a b . \sqrt{a^2 + b^2}) 3 \enspace 4 \]

This is almost all there is to lambda calculus! The details will become clear
as we build our interpreter. Afterwards, we'll see how to compute anything
with it.

I was surprised this substitution process learned in childhood is all we need
for computing anything. A Turing machine has states, a tape of cells, and a
movable head that reads and writes; how can putting formulas into formulas be
equivalent? [In retrospect, maybe my surprise was unwarranted.
https://en.wikipedia.org/wiki/Tag_system[Tag systems] are Turing-complete, as
is https://en.wikipedia.org/wiki/Conway's_Game_of_Life[Conway's Game of
Life].]

To build everything yourself, install http://haste-lang.org/[Haste] and
http://asciidoc.org[AsciiDoc], and then type:

------------------------------------------------------------------------------
$ haste-cabal install parsec
$ wget https://crypto.stanford.edu/~blynn/haskell/lambda.lhs
$ hastec lambda.lhs
$ sed 's/^\\.*{code}$/-----/' lambda.lhs | asciidoc -o - - > lambda.html
$ cabal install parsec readline
$ ghc lambda.lhs
------------------------------------------------------------------------------

Then run the command-line interpreter `./lambda` or browse to `lambda.html`.

To produce binaries for different systems, we need conditional compilation
and various imports:

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

== Terms ==

'Lambda calculus terms' can be viewed as a kind of full binary tree. A lambda
calculus term consists of:

  * 'Variables', which we can think of as leaf nodes holding strings.
  * 'Applications', which we can think of as internal nodes.
  * 'Lambda abstractions', which we can think of as a special kind of internal
  node whose left child must be a variable.

\begin{code}
data Term = Var String | App Term Term | Lam String Term
\end{code}

When printing terms, we'll use Unicode to show a lowercase lambda (&#0955;).
Conventionally:

  * Function application has higher precedence, associates to the left, and
    their child nodes are juxtaposed.
  * Lambda abstractions associate to the right, are prefixed with a lowercase
    lambda, and their child nodes are separated by periods. The lambda prefix
    is superfluous but improves clarity.
  * With consecutive bindings (e.g. "λx0.λx1...λxn."), we omit all lambdas but
    the first, and omit all periods but the last (e.g. "λx0 x1 ... xn.").

For clarity, we enclose lambdas in parentheses if they are right child of an
application.

\begin{code}
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

As for input, since typing Greek letters can be nontrivial, we follow Haskell
and interpret the backslash as lambda. We may as well follow Haskell a little
further and accept `->` in lieu of periods, and support line comments.

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

Our parser accepts empty lines, which should be ignored by the interpreter.

\begin{code}
data LambdaLine = Let String Term | Run Term | Empty

line :: Parser LambdaLine
line = (((eof >>) . pure) =<<) . (ws >>) $ option Empty $ do
  t <- term
  option (Run t) $ str "=" >> Let (getV t) <$> term where
  getV (Var s) = s
  term = lam <|> app
  lam = flip (foldr Lam) <$> between lam0 lam1 (many1 v) <*> term where
    lam0 = str "\\" <|> str "\0955"
    lam1 = str "->" <|> str "."
  app = foldl1' App <$> many1 ((Var <$> v) <|> between (str "(") (str ")") term)
  v   = many1 alphaNum >>= (ws >>) . pure
  str = (>> ws) . string
  ws = spaces >> optional (try $ string "--" >> many anyChar)
\end{code}

== Evaluation ==

If the root node is a free variable or a lambda, then there is nothing to do.
Otherwise, the root node is an App node, and we recursively evaluate the left
child.

If the left child evaluates to anything but a lambda, then we stop, as a free
variable got in the way somewhere.

Otherwise, we perform beta reduction as follows. Let the left child be $\lambda
v . M$. We traverse the right subtree of the root node, and replace every
occurrence of $v$ with the term $M$.

While doing so, we must handle a  potential complication. A reduction such as
`(\y -> \x -> y)x` to `\x -> x` is incorrect. To prevent this, we rename the
first occurence of `x` to get `\x1 -> x`.

More precisely, a variable `v` is 'bound' if it appears in the right subtree of
a lambda abstraction node whose left child is `v`. Otherwise `v` is 'free'. If a
substitution would cause a free variable to become bound, then we rename all
free occurrences of that variable before proceeding. The new name must differ
from all other free variables.

We store the let definitions in an associative list named `env`, and perform
lookups on demand to see if a given string is a variable or shorthand for
another term.

We're supposed to expand all such definitions before beginning evaluation, so
that we can be sure the input is a valid term, but this lazy approach enables
recursive let definitions. Thus our interpreter actually runs more than plain
lambda calculus, in the same way that Haskell allows unrestricted recursion
tacked on a typed variant of lambda calculus.

The first line is a special feature that will be explained later.

\begin{code}
eval env (App (Var "encode") t) = encode env t
eval env term@(App m a) | Lam v f <- eval env m   = let
  beta (Var s)   | s == v         = a
                 | otherwise      = Var s
  beta (Lam s m) | s == v         = Lam s m
                 | s `elem` fvs   = let s1 = newName s fvs in
                   Lam s1 $ beta $ rename s s1 m
                 | otherwise      = Lam s (beta m)
  beta (App m n)                  = App (beta m) (beta n)
  fvs = fv env [] a
  in eval env $ beta f
eval env term@(Var v)   | Just x  <- lookup v env = eval env x
eval _   term                                     = term

fv env vs (Var s) | s `elem` vs            = []
                  | Just x <- lookup s env = fv env (s:vs) x
                  | otherwise              = [s]
fv env vs (App x y)                        = fv env vs x `union` fv env vs y
fv env vs (Lam s f)                        = fv env (s:vs) f
\end{code}

To pick a new name, we increment the number at the end of the name (or append
"1" if there is no number) until we've avoided all the given names.

\begin{code}
newName x ys = head $ filter (`notElem` ys) $ (s ++) . show <$> [1..] where
  s = dropWhileEnd isDigit x
\end{code}

Renaming a free variable is a tree traversal that skips lambda abstractions
that bind them:

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
(beta reductions) are possible. We recursively call `eval` on child nodes to
reduce other function applications throughout the tree, resulting in the
'normal form' of the lambda term. The normal form is unique up to variable
renaming (which is called 'alpha-conversion').

\begin{code}
norm env term = case eval env term of
  Var v   -> Var v
  Lam v m -> Lam v (rec m)
  App m n -> App (rec m) (rec n)
  where rec = norm env
\end{code}

A term with no free variables is called a 'closed lambda expression' or
'combinator'. When given such a term, our function's output contains no `App`
nodes.

That is, if it ever outputs something. There's no guarantee that our recursion
terminates. For example, it is impossible to reduce all the `App` nodes of:

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

Lastly, the user interface:

\begin{code}
#ifdef __HASTE__
main = withElems ["input", "output", "evalB",
                  "factB", "factP", "surB", "surP"] $
  \[iEl, oEl, evalB, factB, factP, surB, surP] -> do
  factB `onEvent` Click $ const $
    getProp factP "value" >>= setProp iEl "value" >> setProp oEl "value" ""
  surB `onEvent` Click $ const $
    getProp surP "value" >>= setProp iEl "value" >> setProp oEl "value" ""
  evalB `onEvent` Click $ const $ do
    let
      run (out, env) (Left err) =
        (out ++ "parse error: " ++ show err ++ "\n", env)
      run (out, env) (Right m) = case m of
        Empty      -> (out, env)
        Run term   -> (out ++ show (norm env term) ++ "\n", env)
        Let s term -> (out, (s, term):env)
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
        Right Empty -> repl env
        Right (Run term) -> do
          print $ norm env term
          repl env
        Right (Let s term) -> repl ((s, term):env)

main = repl []
#endif
\end{code}

== A Lesson Learned ==

Until I wrote an interpreter, my understanding of renaming was flawed. I knew
that we compute with closed labmda expressions, that is, terms with no free
variables, so I had thought this meant I could skip implementing renaming. No
free variables can become bound because they're all bound to begin with, right?

In an early version of this interpreter, I tried to normalize:

------------------------------------------------------------------------------
(\f x -> f x)(\f x -> f x)
------------------------------------------------------------------------------

My old program mistakenly returned:

------------------------------------------------------------------------------
\x x -> x x
------------------------------------------------------------------------------

It's probably obvious to others, but it was only at this point I realized that
the recursive nature of beta reductions implies that in the right subtree of a
lambda abstraction, a variable may be free, even though it is bound when the
entire tree is considered. With renaming, my program gave the correct answer:

------------------------------------------------------------------------------
\x x1 -> x x1
------------------------------------------------------------------------------

== Computing with lambda calculus ==

When starting out with lambda calculus, we soon miss the symbols of Turing
machines. We endlessly substitute functions in other functions. They never
``bottom out''. Apart from punctuation, we only see a soup of variable names
and lambdas. No numbers nor arithmetic operations. Even computing 1 + 1 seems
impossible!

The trick is to use functions to represent data. This is less intuitive than
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

Integers can be encoded in a unary manner:

------------------------------------------------------------------------------
0 = \f x -> x
1 = \f x -> f x
2 = \f x -> f (f x)
-- ...and so on.
------------------------------------------------------------------------------

We can perform arithmetic on them with the following:

------------------------------------------------------------------------------
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

The predecessor function is far slower than the successor function, as it
constructs the answer by starting from 0 and repeatedly computing the successor.
There is no quick way to ``strip off'' one layer of a function application.

We can pair up any two terms as follows:

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

Instead of unary, we could encode numbers in binary by using lists of booleans.
This is of course more efficient, but then we lose the elegant spartan
equations for arithmetic that remind us of the Peano axioms.

== Recursion ==

Because our interpreter cheats and only looks up a let definition at the last
minute, we can recursively compute factorials with:

------------------------------------------------------------------------------
factrec = \n -> if (is0 n) 1 (mul n (factrec (pred n)))
------------------------------------------------------------------------------

But we stress this is not a lambda calculus term. If we tried to expand the let
definitions, we'd be forever replacing `factrec` with an expression containing
a `factrec`. We'd never eliminate all the function names and reach a valid
lambda calculus term.

Instead, we need something like the
https://en.wikipedia.org/wiki/Fixed-point_combinator['Y combinator']. The inner
workings are described in many other places, so we'll content ourselves
with listing their definitions, and observing they are indeed lambda calculus
terms.

------------------------------------------------------------------------------
Y = \f -> (\x -> f(x x))(\x -> f(x x))
fact = Y(\f n -> if (is0 n) 1 (mul n (f (pred n))))
------------------------------------------------------------------------------

Thus we can simulate any Turing machine with a lambda calculus term: we could
concoct a data structure to represent a tape, which we'd feed into a recursive
function that carries out the state transitions.

[pass]
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
<textarea id="factP" hidden>
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
fact (succ (succ (succ 1)))  -- Compute 4!
</textarea>
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

== Lambda calculus with lambda calculus ==

http://repository.readscheme.org/ftp/papers/topps/D-128.pdf[Mogensen
describes a delightful encoding of lambda terms with lambda terms]. If
we denote the encoding of a term $T$ by $\lceil T\rceil$,
then we can recursively encode any term with the following three rules for
variables, applications, and lambda abstractions, respectively:

\[
\begin{align}
\lceil x \rceil &= \lambda a b c . a x \\
\lceil M N \rceil &= \lambda a b c . b \lceil M\rceil \lceil N\rceil \\
\lceil \lambda{x}.M \rceil &= \lambda a b c . c (\lambda x . \lceil M\rceil)
\end{align}
\]

where $a, b, c$ may be renamed to avoid clashing with any free variables in
the term being encoded. In our code, this translates to:

\begin{code}
encode env t = case t of
  Var x   | Just t <- lookup x env -> rec t
          | otherwise              -> f 0 (\v -> App v $ Var x)
  App m n                          -> f 1 (\v -> App (App v (rec m)) (rec n))
  Lam x m                          -> f 2 (\v -> App v $ Lam x $ rec m)
  where
    rec = encode env
    fvs = fv env [] t
    f n g = Lam a (Lam b (Lam c (g $ Var $ abc!!n)))
    abc@[a, b, c] = renameIfNeeded <$> ["a", "b", "c"]
    renameIfNeeded s | s `elem` fvs = newName s fvs
                     | otherwise    = s
\end{code}

With this encoding the following lambda term `E` is a self-interpreter,
that is, $E \lceil M \rceil$ evaluates to the normal form of $M$ if it exists:

------------------------------------------------------------------------------
E = Y(\e m -> m (\x -> x) (\m n -> (e m)(e n)) (\m v -> e (m v)))
------------------------------------------------------------------------------

Also, the following lambda term `R` is a self-reducer, which means
$R \lceil M \rceil$ evaluates to the encoding of the normal form of $M$ if
it exists:

------------------------------------------------------------------------------
P = Y(\p m -> (\x -> x(\v -> p(\a b c -> b m(v (\a b -> b))))m))
RR = Y(\r m -> m (\x -> x) (\m n -> (r m) (\a b -> a) (r n)) (\m -> (\g x -> x g(\a b c -> c(\w -> g(P (\a b c -> a w))(\a b -> b)))) (\v -> r(m v) )))
R = \m -> RR m (\a b -> b)
------------------------------------------------------------------------------

Unlike the self-interpreter, the self-reducer requires the input to be the
encoding of a closed term. See Mogensen's paper for details.

[pass]
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
<textarea id="surP" hidden>-- See Mogensen, "Efficient Self-Interpretation in Lambda Calculus".
Y = \f -> (\x -> f(x x))(\x -> f(x x))
E = Y(\e m -> m (\x -> x) (\m n -> (e m)(e n)) (\m v -> e (m v)))
P = Y(\p m -> (\x -> x(\v -> p(\a b c -> b m(v (\a b -> b))))m))
RR = Y(\r m -> m (\x -> x) (\m n -> (r m) (\a b -> a) (r n)) (\m -> (\g x -> x g(\a b c -> c(\w -> g(P (\a b c -> a w))(\a b -> b)))) (\v -> r(m v) )))
R = \m -> RR m (\a b -> b)
1 = \f x -> f x
succ = \n f x -> f(n f x)
E (encode (succ (succ (succ 1))))  -- Self-interpreter demo.
R (encode (succ (succ (succ 1))))  -- Self-reducer demo.
</textarea>
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

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

This seems quotidian so far. Typical high-level languages do this sort
of thing. The fun part is seeing how easily it can be tacked on to lambda
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

In other words, programs always halt. If our interpreter cheats to allow
recursion, then this is no longer true, but at least it narrows down the
suspect parts of our program; furthermore, by restricting recursion in certain
ways, we can regain the assurance that our programs will halt.

Try doing this with Turing machines!

== Practical lambda calculus ==

The above factorial functions are shorter than equivalent code in many
high-level languages. Indeed, unlike Turing machines, we can turn lambda
calculus into a practical programming langauge with just a few tweaks.

We've already seen one such tweak: we can allow recursion by expanding a
let definition on demand.

But we must overcome a giant obstacle if we wish to program with lambda
calculus: real programs have side effects. After all, why bother computing a
number if there's no way to print it?

The most tempting solution is to allow functions to have side effects, for
example, functions may print to the screen when evaluated. This requires us to
carefully specify exactly when a function is evaluated.

link:lisp.html[Lisp does this by stipulating applicative order], so we can
reason about the ordering of side effects, and provides a macro feature to
override applicative order for special cases. Unfortunately, we lose nice
features of the theory: notably, some programs that would halt with a
normal-order evaluation strategy will loop forever.

It turns out there are other solutions that keep functions pure and hence
stay true to theory. Haskell chose one such solution. As a result, normal-order
evaluation works in Haskell.

Additionally, Haskell is built on an advanced typed lambda calculus that is
expressive yet strongly normalizing (though unrestricted recursion means
programs can loop forever).

We've only scratched the surface. Lambda calculus is but one interesting node
in a giant tree that includes combinatory logic, type inference, richer type
systems, provably correct programs, and more.
