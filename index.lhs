= Lambda Calculus =

They taught us Turing machines in my computer science classes.
Despite being purely theoretical, Turing machines are important:

 - A state machine reading and writing symbols on an infinite tape is a
 useful abstraction of a CPU reading from and writing to RAM.
 - Many high-level programming languages adhere to the same model: code writes
   data to memory, then later reads it to decide a future course of action.
 - We immediately see how to measure algorithmic complexity.
 - Encoding a Turing machine on a tape is straightforward, and gently guides
 us to the equivalence of code and data, with all its deep implications.

All the same, I wish they had given equal weight to an alternative model of
computation known as https://en.wikipedia.org/wiki/Lambda_calculus['lambda calculus']:

[pass]
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
<script src="index.js"></script>
<p>
<button id="expB">2^3</button>
<button id="factB">Factorial</button>
<button id="quoteB">Quote</button>
<button id="surB">Surprise Me!</button>
<button id="shaB">Shallow</button></p>
<p><textarea style="border: solid 2px; border-color: #999999" id="input" rows="12" cols="80">
</textarea></p>
<button id="evalB">Run</button>
<p><textarea id="output" rows="3" cols="80" readonly></textarea></p>
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Also see
https://projectultimatum.org/cgi-bin/lambda[Hai (Paul) Liu's step-by-step Lambda Viewer].

== Why Lambda Calculus? ==

Lambda calculus is historically significant. Alonzo Church was Alan Turing's
doctoral advisor, and his lambda calculus predates Turing machines.

But more importantly, working through the theory from its original viewpoint
exposes us to different ways of thinking. Aside from a healthy mental workout,
we find lambda calculus is sometimes superior:

 * *simple*: Here's how to multiply two numbers in lambda calculus:
 $\lambda m.\lambda n.\lambda f.m(n f)$. Spare a thought for
 https://stackoverflow.com/questions/tagged/turing-machines[students struggling
 to make Turing machines do simple tasks].

 * *practical*: With a little syntax sugar, lambda calculus becomes
 a practical programming language. Already, our factorial example above is
 shorter than equivalent code in many high-level languages!
 In contrast, sweetened Turing machines would probably still be unpalatable.

 * *one-line universal program*: Here's a lambda calculus self-interpreter:
 $(\lambda f.(\lambda x.f(xx))(\lambda x.f(xx)))(\lambda em.m(\lambda x.x)(\lambda mn.em(en))(\lambda mv.e(mv)))$.
 In contrast, universal Turing machines are so tedious that classes often
 skip the details and just explain why they exist.

 * *representing data with functions* leads to rich algebras where a
 little notation goes a long way. For example, we can
 https://diagrams.github.io/gallery.html[draw intricate diagrams
 with a few lines].

 * link:simply.html[*solves the halting problem*]: By adding types, we can
 ensure lambda calculus programs always halt. It's unclear how we can
 similarly tame Turing machines.

 * *provably correct*: More generally,
 https://en.wikipedia.org/wiki/Curry%E2%80%93Howard_correspondence[typed lambda
 calculus turns out to be deeply connected to the foundations of mathematics].
 Sufficiently advanced types make bugs impossible to express, that is, every
 syntactically correct program is also semantically correct. This connection is
 harder to see from a Turing-machine viewpoint.

As the importance of software grows in our world, so does the importance of
the advantages of lambda calculus, and in particular, its connections with the
foundations of mathematics. Computer science without lambda calculus is like
engineering without physics.

A better analogy would be mathematics without proofs, but proofs have been part
of mathematics for so long that it may be difficult to imagine one without the
other.

== Why Lambda? ==

See
https://www-users.mat.umk.pl/\~adwid/materialy/doc/church.pdf['The impact of lambda calculus
in logic and computer science'] by Henk Barendregt, and
https://pdfs.semanticscholar.org/959d/32cfa6df9299312ba51e2102045e1f25bc18.pdf['History of Lambda-calculus and Combinatory Logic'] by Felice Cardone and
J. Roger Hindley. It seems its true name should be ``hat calculus''.
We'll find that lambdas are redundant, but I suppose we need a symbol of some sort
to avoid calling it just ``calculus''.

== Beta reduction ==

Unlike Turing machines, everyone already knows the basics of lambda calculus.
In school, we're accustomed to evaluating functions. In fact, one might argue
they focus too much on making students memorize and apply formulas such as
$\sqrt{a^2 + b^2}$ for $a = 3$ and $b = 4$.

In lambda calculus, this is called 'beta reduction', and we'd write this
example as:

\[ (\lambda a b . \sqrt{a^2 + b^2}) 3 \enspace 4 \]

This is almost all there is to lambda calculus! Only, instead of numbers,
we plug in other formulas. The details will become clear as we build our
interpreter.

I was surprised this substitution process learned in childhood is all we need
for computing anything. A Turing machine has states, a tape of cells, and a
movable head that reads and writes; how can putting formulas into formulas be
equivalent?

We use code to help answer the question, which requires a bit of boilerplate:

\begin{code}
{-# LANGUAGE CPP #-}
#ifdef __HASTE__
import Haste.DOM
import Haste.Events
#else
import System.Console.Haskeline
#endif
import Data.Char
import Data.List
import Text.Parsec
\end{code}

== Terms ==

'Lambda calculus terms' can be viewed as a kind of binary tree. A lambda
calculus term consists of:

  * 'Variables', which we can think of as leaf nodes holding strings.
  * 'Applications', which we can think of as internal nodes.
  * 'Lambda abstractions', which we can think of as a special kind of internal
  node whose left child must be a variable. (Or as a internal node labeled
  with a variable with exactly one child.)

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
  * With consecutive bindings (e.g. "λx.λy.λz."), we omit all lambdas but
    the first, and omit all periods but the last (e.g. "λx y z.").

For clarity, we enclose lambdas in parentheses if they are right child of an
application.

\begin{code}
instance Show Term where
  show (Lam s t)  = "\0955" ++ s ++ showB t where
    showB (Lam x y) = " " ++ x ++ showB y
    showB expr      = "." ++ show expr
  show (Var s)    = s
  show (App x y)  = showL x ++ showR y where
    showL (Lam _ _) = "(" ++ show x ++ ")"
    showL _         = show x
    showR (Var s)   = ' ':s
    showR _         = "(" ++ show y ++ ")"
\end{code}

As for input, since typing Greek letters can be nontrivial, we follow Haskell
and interpret the backslash as lambda. We may as well follow Haskell a little
further and accept `->` in lieu of periods, and support line comments.

Any alphanumeric string is a valid variable name.

Typing a long term is laborious, so we support a sort of 'let' statement. The
line

------------------------------------------------------------------------------
true = \x y -> x
------------------------------------------------------------------------------

means that for all following terms, the variable `true` is shorthand for the
term on the right side, namely `\x y -> x`.
There is an exception: if the left child of a lambda abstraction is the
variable `true`, then this variable 'shadows' the original let definition in
its right child. It is good style to avoid such name collisions.

Our parser accepts empty lines, let definitions, or terms that should be
evaluated immediately.

\begin{code}
data LambdaLine = Blank | Let String Term | Run Term

line :: Parsec String () LambdaLine
line = between ws eof $ option Blank $
    try (Let <$> v <*> (str "=" >> term)) <|> (Run <$> term) where
  term = lam <|> app
  lam = flip (foldr Lam) <$> between lam0 lam1 (many1 v) <*> term
  lam0 = str "\\" <|> str "\0955"
  lam1 = str "->" <|> str "."
  app = foldl1' App <$> many1 ((Var <$> v) <|> between (str "(") (str ")") term)
  v   = many1 alphaNum <* ws
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

While doing so, we must handle a potential complication. A reduction such as
`(\y -> \x -> y)x` to `\x -> x` is incorrect. To prevent this, we rename the
first `x` and find `(\y -> \x1 -> y)x` reduces to `\x1 -> x`.

More precisely, a variable `v` is 'bound' if it appears in the right subtree of
a lambda abstraction node whose left child is `v`. Otherwise `v` is 'free'. If a
substitution would cause a free variable to become bound, then we rename all
free occurrences of that variable before proceeding. The new name must differ
from all other free variables.

We store the let definitions in an associative list named `env`, and perform
lookups on demand to see if a given string is a variable or shorthand for
another term.

These on-demand lookups and the way we update `env` means recursive let
definitions are possible. Thus our interpreter actually runs more than plain
lambda calculus; a true lambda calculus term is unable to refer to itself.
(Haskell similarly permits unrestricted recursion via let expressions.)

The quote business is a special feature that will be explained later.

\begin{code}
eval env (App (Var "quote") t)              = quote env t
eval env (App m a) | Lam v f <- eval env m  =
  eval env $ beta env (v, a) f
eval env (Var v)   | Just x <- lookup v env = eval env x
eval _   term                               = term

beta env (v, a) t = case t of
  Var s   | s == v       -> a
          | otherwise    -> Var s
  Lam s m | s == v       -> Lam s m
          | s `elem` fvs -> Lam s1 $ rec $ rename s s1 m
          | otherwise    -> Lam s (rec m)
          where s1 = newName s $ fvs `union` fv env [] m
  App m n                -> App (rec m) (rec n)
  where rec = beta env (v, a)
        fvs = fv env [] a

fv env vs (Var s) | s `elem` vs            = []
                  -- Handle free variables in let definitions.
                  -- Avoid repeatedly following recursive lets.
                  | Just x <- lookup s env = fv env (s:vs) x
                  | otherwise              = [s]
fv env vs (App x y)                        = fv env vs x `union` fv env vs y
fv env vs (Lam s f)                        = fv env (s:vs) f
\end{code}

To pick a new name, we increment the number at the end of the name (or append
"1" if there is no number) until we've avoided all the given names.

Our simplistic scheme relies on the user to avoid, say, using the name "x1"
within a lambda abstraction that binds the variable "x". We could solve this
problem by also prepending some character that our parser rejects as part of
a variable name, but it makes our interactive demo uglier.

\begin{code}
newName :: String -> [String] -> String
newName x ys = head $ filter (`notElem` ys) $ (s ++) . show <$> [1..] where
  s = dropWhileEnd isDigit x
\end{code}

Renaming a free variable is a tree traversal that skips lambda abstractions
that bind them:

\begin{code}
rename :: String -> String -> Term -> Term
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
norm :: [(String, Term)] -> Term -> Term
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

In an application `App m n`, the function `eval` tries to reduce `m` first.
This is called a 'normal-order evaluation strategy'.
What if we reduced `n` first, a strategy known as 'applicative order'?
More generally, instead of starting at the top
level, what if we picked some sub-expression to reduce first? Does it matter?

Yes and no. On the one hand, the
https://en.wikipedia.org/wiki/Church%E2%80%93Rosser_theorem['Church-Rosser
theorem'] states that the order of evaluation is unimportant in that if terms
$b$ and $c$ are both derived from term $a$, then there exists a term $d$ to
which both $b$ and $c$ can be reduced. In particular, if we reach a term where
no further reductions are possible, then it must be the normal form we defined
above.

On the other hand, some strategies may loop forever instead of normalizing a
term that does in fact possess a normal form. It turns out this never happens
with normal-order evaluation: it always reduces a term to its normal form if it
exists, hence its name.

This is intuitively evident, as at each step we're doing the bare minimum.
Reducing `m` before `n` means we ignore arguments to a function until they are
needed, which explains another name for this strategy: 'lazy evaluation'.
Technically, this is a slightly different strategy because it also means we are
memoizing to avoid repeating computations, but this only matters when we're
considering efficiency.

== User interface ==

We've saved the worst for last:

\begin{code}
#ifdef __HASTE__
main = withElems ["input", "output", "evalB"] $ \[iEl, oEl, evalB] -> do
  let
    prep s = do
      Just button <- elemById $ s ++ "B"
      button `onEvent` Click $ const $ setInput s
    setInput s = do
      Just para   <- elemById $ s ++ "P"
      getProp para "value" >>= setProp iEl "value" >> setProp oEl "value" ""
  setInput "exp"
  mapM_ prep $ words "exp fact quote sur sha"
  evalB `onEvent` Click $ const $ do
    let
      run (out, env) (Left err) =
        (out ++ "parse error: " ++ show err ++ "\n", env)
      run (out, env) (Right m) = case m of
        Blank      -> (out, env)
        Run term   -> (out ++ show (norm env term) ++ "\n", env)
        Let s term -> (out, (s, term):env)
    es <- map (parse line "") . lines <$> getProp iEl "value"
    setProp oEl "value" $ fst $ foldl' run ("", []) es
#else
repl env = do
  ms <- getInputLine "> "
  case ms of
    Nothing -> outputStrLn ""
    Just s  -> case parse line "" s of
      Left err  -> do
        outputStrLn $ "parse error: " ++ show err
        repl env
      Right Blank -> repl env
      Right (Run term) -> do
        outputStrLn $ show $ norm env term
        repl env
      Right (Let v term) -> repl ((v, term):env)

main = runInputT defaultSettings $ repl []
#endif
\end{code}

== A Lesson Learned ==

Until I wrote an interpreter, my understanding of renaming was flawed. I knew
that we compute with closed lambda expressions, that is, terms with no free
variables, so I had thought this meant I could ignore renaming. No free
variables can become bound because they're all bound to begin with, right?

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

== Booleans, Numbers, Pairs ==

When starting out with lambda calculus, we soon miss the symbols of Turing
machines. We endlessly substitute functions in other functions. They never
bottom out. Apart from punctuation, we only see a soup of variable names
and lambdas. No numbers nor arithmetic operations. Even computing 1 + 1 seems
impossible!

The trick is to use functions to represent data. This is less intuitive than
encoding Turing machines on a tape, but well worth learning. The original and
most famous scheme is known as 'Church encoding'. We'll only summarize briefly.
See:

 * https://www.reddit.com/r/haskell/comments/5vbvul/keynote_why_functional_programming_matters_john/[``Why
functional programming matters'', a keynote address by John Hughes and Mary
Sheeran].
 * https://www.cs.yale.edu/homes/hudak/CS201S08/lambda.pdf['A Brief and Informal
Introduction to the Lambda Calculus'] by Paul Hudak.
 * https://en.wikipedia.org/wiki/Church_encoding[Wikipedia's entry on Church
encoding].

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
There is no quick way to strip off one layer of a function application.

We can pair up any two terms as follows:

------------------------------------------------------------------------------
pair = \x y z -> z x y
fst = \p -> p true
snd = \p -> p false
------------------------------------------------------------------------------

From such tuples, we can construct lists, trees, and so on.

Admittedly, the predecessor function is complicated, probably more so than the
a typical Turing machine implementation. However, this is an artifact of the
Church encoding. With
https://ifl2014.github.io/submissions/ifl2014_submission_13.pdf[the Scott
encoding], we have a fast and simple predecessor function:

------------------------------------------------------------------------------
0 = \f x -> x
succ = \n f x -> f n
pred = \n -> n (\x -> x) 0
is0 = \n -> n (\x -> false) true
------------------------------------------------------------------------------

Instead of unary, we could encode numbers in binary by using lists of booleans.
Though more efficient, we lose the elegant spartan equations for arithmetic
that remind us of the Peano axioms.

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
with definitions, and observing they are indeed lambda calculus terms.

------------------------------------------------------------------------------
Y = \f -> (\x -> f(x x))(\x -> f(x x))
fact = Y(\f n -> if (is0 n) 1 (mul n (f (pred n))))
------------------------------------------------------------------------------

Thus we can simulate any Turing machine with a lambda calculus term: we could
concoct a data structure to represent a tape, which we'd feed into a recursive
function that carries out the state transitions.

[pass]
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
<textarea id="expP" hidden>
2 = \f x -> f (f x)
3 = \f x -> f (f (f x))
exp = \m n -> n m
exp 2 3  -- Compute 2^3.
</textarea>
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

https://www.researchgate.net/publication/2673832_Efficient_Self-Interpretation_in_Lambda_Calculus[Mogensen
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
quote env term = case term of
  Var x   | Just t <- lookup x env -> rec t
          | otherwise              -> f 0 (\v -> App v $ Var x)
  App m n                          -> f 1 (\v -> App (App v (rec m)) (rec n))
  Lam x m                          -> f 2 (\v -> App v $ Lam x $ rec m)
  where
    rec = quote env
    fvs = fv env [] term
    f n g = Lam a (Lam b (Lam c (g $ Var $ abc!!n)))
    abc@[a, b, c] = renameIfNeeded <$> ["a", "b", "c"]
    renameIfNeeded s | s `elem` fvs = newName s fvs
                     | otherwise    = s
\end{code}

With this encoding the following lambda term is a self-interpreter,
that is, $E \lceil M \rceil$ evaluates to the normal form of $M$ if it exists:

------------------------------------------------------------------------------
E = Y(\e m.m (\x.x) (\m n.(e m)(e n)) (\m v.e (m v)))
------------------------------------------------------------------------------

Also, the following lambda term `R` is a self-reducer, which means
$R \lceil M \rceil$ evaluates to the encoding of the normal form of $M$ if
it exists:

------------------------------------------------------------------------------
P = Y(\p m.(\x.x(\v.p(\a b c.b m(v (\a b.b))))m))
RR = Y(\r m.m (\x.x) (\m n.(r m) (\a b.a) (r n)) (\m.(\g x.x g(\a b c.c(\w.g(P (\a b c.a w))(\a b.b)))) (\v.r(m v))))
R = \m.RR m (\a b.b)
------------------------------------------------------------------------------

Unlike the self-interpreter, the self-reducer requires the input to be the
encoding of a closed term. See Mogensen's paper for details.

[pass]
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
<textarea id="surP" hidden>-- See Mogensen, "Efficient Self-Interpretation in Lambda Calculus".
Y = \f.(\x.f(x x))(\x.f(x x))
E = Y(\e m.m (\x.x) (\m n.(e m)(e n)) (\m v.e (m v)))
P = Y(\p m.(\x.x(\v.p(\a b c.b m(v (\a b.b))))m))
RR = Y(\r m.m (\x.x) (\m n.(r m) (\a b.a) (r n)) (\m.(\g x.x g(\a b c.c(\w.g(P (\a b c.a w))(\a b.b)))) (\v.r(m v))))
R = \m.RR m (\a b.b)
1 = \f x.f x
succ = \n f x.f(n f x)
E (quote (succ (succ (succ 1))))  -- Self-interpreter demo.
R (quote (succ (succ (succ 1))))  -- Self-reducer demo.
</textarea>
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

== Scary quotes ==

Why did we step outside lambda calculus and hard-code the implementation
of `quote`? Maybe we can define it within lambda calculus.
Let's suppose so. Then consider the expression:

------------------------------------------------------------------------------
quote ((\y.y) x)
------------------------------------------------------------------------------

If we evaluate the outermost function application, we get:

------------------------------------------------------------------------------
λa b c.b(λa b c.c(λy a b c.a y))(λa b c.a x)
------------------------------------------------------------------------------

On the other hand, if we first evaluate the sub-expression `((\y.y) x)`, then
it reduces to `quote x`, which is:

------------------------------------------------------------------------------
λa b c.a x
------------------------------------------------------------------------------

This violates the Church-Rosser theorem. In short, `id x = x` but `quote(id
x) /= quote x`. Thus `quote` is not a function, and should be seen as a sort of
macro; a labour-saving abbreviation lying outside of lambda calculus.

We named `quote` after a similar primitive in link:lisp.html[the Lisp
language], which suffers from the same affliction.
The Right Way to reify is to sacrifice brevity:

------------------------------------------------------------------------------
Var=\m.\a b c.a m
App=\m n.\a b c.b m n
Lam=\f.\a b c.c f
------------------------------------------------------------------------------

Then `quote ((\y.y) x)` can be expressed in pure lambda calculus as:

------------------------------------------------------------------------------
App (Lam (\y.Var y)) (Var x)
------------------------------------------------------------------------------

This is less convenient, but it's more comprehensible than the raw encoding.
Most importantly, we're back on firm theoretical ground.

[pass]
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
<textarea id="quoteP" hidden>
Var=\m.\a b c.a m
App=\m n.\a b c.b m n
Lam=\f.\a b c.c f

-- Our `quote` primitive lies outside lambda calculus:
quote ((\y.y) x)

-- Encoding terms in lambda calculus (without typing the raw encoding):
App (Lam (\y.Var y)) (Var x)
</textarea>
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

== A shallow encoding ==

In 'Linear-Time Self-Interpretation of the Pure Lambda Calculus', Mogensen
describes a shorter self-interpreter: `E=\q.q(\x.x)(\x.x)`.

To encode, we pick two unused variables, say `a` and `b`, and prepend `\a b.`
to the term. Then we replace each application `m n` with `a m n` and
each lambda `\x.m` with `b(\x.m)`.

In http://compilers.cs.ucla.edu/popl16/popl16-full.pdf['Breaking Through the
Normalization Barrier: A Self-Interpreter for F-omega'], Matt Brown and Jens
Palsberg describe perhaps the shortest possible self-interpreter:
`E=\q.q(\x.x)`.

To encode a term for this self-interpreter, we pick a new variable, say `i`,
and prepend `\i.` to the term. Then we replace each application `m n` with `i
m n`.

For example, the term `succ 0`, that is, `(\n f x.f(n f x))(\f x.x)`,
becomes:

------------------------------------------------------------------------------
\i.i(\n f x.i f(i(i n f)x))(\f x.x)
------------------------------------------------------------------------------

We can avoid threading the `i` variable throughout the encoding by inserting
another layer of abstraction, so we can reuse our notation above:

------------------------------------------------------------------------------
Var=\x.\i.x
App=\m n.\i.i(i m i)(i n i)
Lam=\f.\i x.f x i
E=\q.q(\x.x)  -- A self-interpreter.
------------------------------------------------------------------------------

This encoding is shallow in the sense that only a self-interpreter can do
anything useful with it.

[pass]
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
<textarea id="shaP" hidden>
-- Shallow encoding.
Var=\x.\i.x
App=\m n.\i.i(m i)(n i)
Lam=\f.\i x.f x i
E=\q.q(\x.x)  -- A self-interpreter.

qsucc = Lam(\n.Lam(\f.Lam(\x.App(Var f)(App(App(Var n)(Var f))(Var x)))))
q0 = Lam(\f.Lam(\x.Var x))
E(App qsucc (App qsucc q0))  -- Compute `succ (succ 0)`.
</textarea>
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
