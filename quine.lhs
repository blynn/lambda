= Quines =

A quine is a self-replicating program, that is, a quine prints itself when run.
If you've never written a quine, I strongly urge you to try to do so before
reading on, so you can collect a large reward.

== Nothing works ==

Some systems accept the empty program as instructions to do nothing before
terminating, thus fulfilling the definition of a quine. For example, on Linux:

------------------------------------------------------------------------
$ touch nothing
$ chmod +x nothing
$ ./nothing
$
------------------------------------------------------------------------

The `smr` entry of the https://www.ioccc.org/years-spoiler.html#1994[1994
International Obfuscated C Code Contest] won using this principle. Some C
compilers accept the empty input and produce an empty executable which we've
just seen prints the empty string. The IOCCC rules now forbid empty programs,
as do we.

Another easy solution is to use a built-in instruction that prints the program,
available in languages such as BASIC:

------------------------------------------------------------------------
10 LIST
------------------------------------------------------------------------

Computer viruses often execute instructions to read the memory where they have
been loaded in order to self-replicate, which is partly demonstrated by
link:../asm/boot.html[my boot-sector quine].

Though ingenious, these examples cheat; they satisfy the definition of a quine
only in letter and not in spirit.

== Think inside the box ==

An honest quine is more mathematical. In such a program,
https://www.cs.cmu.edu/~crary/819-f09/Landin66.pdf[as Landin put it]:

________________________________________________________________________
the thing an expression denotes, i.e. its "value", depends on the values of its
subexpressions, not on other properties of them.
________________________________________________________________________

For example, the `LIST` command of BASIC is disqualified because its meaning
differs for different programs. Similarly, the effect of a memory read
instruction depends on the contents of the memory, and not just the read
instruction itself. Contrast this with an expression like `1 + 1`, which always
evaluates to 2 wherever it appears.

Thus we seek a mostly pure expression that evaluates to a string that
represents the original expression. For practical purposes, we'll allow
syscalls that print a character or a string to the output.

== Stringing along ==

With these stricter conditions, a first guess might be to write a single print
statement for the program that prints out the source. This works in, say, the
`echo` language in which every "expression" evaluates to its own
representation, but in general, a single print statement fails because it omits
the call to the print function. For example:

------------------------------------------------------------------------
puts("myself");
------------------------------------------------------------------------

prints `myself`, but not the `puts()` call, nor the quotes.

We could try fixing this by quoting the entire program and passing it to
`puts()`:

------------------------------------------------------------------------
puts("puts(\"myself\")");
------------------------------------------------------------------------

But this prints our first program, and not itself.

Nesting more and more print statements is of no use. We'll always be off by
one.

== Von Neumann ==

https://en.wikipedia.org/wiki/John_von_Neumann[John von Neumann] thought deeply
about self-replication, and solved this turtles-all-the-way-down problem by
https://en.wikipedia.org/wiki/Von_Neumann_universal_constructor[designing a
self-replicating machine with three parts]:

 * a program, which we may think of as a string of bits

 * a 'universal constructor', which can interpret the program, that is, carry
 out the instructions represented by a string of bits

 * a 'universal copy machine', which can duplicate a given string of bits.

The constructor builds the new machine except for its program. The copy machine
duplicates the program.

https://www.scientificamerican.com/article/go-forth-and-replicate-2008-02/[Scientific
American describes a delightful analogy] involving an architect drawing a
blueprint to duplicate their studio, with the catch that the blueprint is part
of the studio: "the blueprint would include a plan for building a photocopy
machine. Once the new studio and the photocopier were built, the construction
crew would simply run off a copy of the blueprint and put it into the new
studio." Then "the self-description need not contain a description of itself".

In other words, separating construction from copying sidesteps the infinite
regress. Discovering this key insight is the reward for writing a quine without
looking up the solution first. You get to feel as smart as von Neumann!
Or at least a little bit: von Neumann had more to worry about, because he
envisioned a self-replicating machine with a capacity to evolve, and worked
through the cellular automata version of the problem.

We programmers have it easy. The print statement is our universal constructor:
it outputs any string we give it. And a universal copy machine is only a little
more work: we just need to declare a given string as a string literal by
following the rules of our language. For example, it may be enough to surround
it with quotation marks.

Haskell has a built-in copy machine: the `show` function. When given a string,
it returns a quoted version of the string, ready for use as a string literal:

------------------------------------------------------------------------
> putStrLn $ show "foo"
"foo"
------------------------------------------------------------------------

This leads to a a simple Haskell quine:

------------------------------------------------------------------------
main = let f s = s ++ show s in putStrLn $ f
  "main = let f s = s ++ show s in putStrLn $ f"
------------------------------------------------------------------------

In the definition of `f`, the first `s` is the constructor building the new
machine, while `show s` is the copier duplicating the blueprint. We concatenate
their outputs with `(++)` because juxtaposition means function application in
Haskell.

The string literal can be considered a program in a simple language which can
only do one thing: print the input string and append a quoted version of it.
This is exactly what our interpreter does.

== Boutique quines ==

For some reason, brevity is often prized in quines, so we might use the Reader
monad to reduce the program's size:

------------------------------------------------------------------------
main=putStrLn$(++)<*>show$"main=putStrLn$(++)<*>show$"
------------------------------------------------------------------------

Unlocking the secret of quines enables the construction of curiosities such as
a quine-like program that prints a different version of itself every run by
incrementing a serial number:

------------------------------------------------------------------------
f (n, p) = unlines $ ("serial = " ++ show n):p ++ [' ':show (n + 1, p)]
main = putStr $ f (0,
 ["f (n, p) = unlines $ (\"serial = \" ++ show n):p ++ [' ':show (n + 1, p)]"
 , "main = putStr $ f"])
------------------------------------------------------------------------

Another goal might be to make the quine less obvious. The following is not
a quine:

------------------------------------------------------------------------
main = let f s = (toEnum <$> s) ++ show s in putStrLn $ f t
t = fromEnum <$> "main = let f s = (toEnum <$> s) ++ show s in putStrLn $ f"
------------------------------------------------------------------------

Howeever, it produces the following program, which is a quine where the string
has been replaced by a list of ASCII codes:

------------------------------------------------------------------------
main = let f s = (toEnum <$> s) ++ show s in putStrLn $ f[109,97,105,110
,32,61,32,108,101,116,32,102,32,115,32,61,32,40,116,111,69,110,117,109,32
,60,36,62,32,115,41,32,43,43,32,115,104,111,119,32,115,32,105,110,32,112
,117,116,83,116,114,76,110,32,36,32,102]
------------------------------------------------------------------------

== Use or Mention? ==

https://en.wikipedia.org/wiki/Willard_Van_Orman_Quine[William Van Orman Quine]
wrote 'Mathematical Logic', based on his graduate teaching in the 1930s and
1940s, where he explains
https://en.wikipedia.org/wiki/Use%E2%80%93mention_distinction[the use-mention
distinction] with examples:

  1. Boston is populous.
  2. Boston is disyllabic.
  3. "Boston" is disyllabic.

In the first two cases, the first word plays an active role (use), while it
plays a passive role (mention) in the third case. This distinction alone makes
the second statement false and the third true.

Quine notes "Frege seems to have been the first logician to recognize the
importance of scrupulous use of quotation marks for avoidance of confusion
between use and mention of expressions", and laments that this lesson was
largley ignored for 30 years.

Today, thanks to the web, even non-programmers may be aware of this
distinction. URLs often appear intelligible except for bits and pieces replaced
by percent signs and numbers. A layman might rightly suspect that this is a
form of quoting which prevents the browser from acting a certain way on certain
characters, that is, a means to indicate a mention and not a use.

== Quine's Paradox ==

Self-replicating programs are called quines because of Quine's paradox:

  * yields falsehood when preceded by its quotation.

If we do what it suggests, we get:

  * "yields falsehood when preceded by its quotation" yields falsehood when
  preceded by its quotation.

which is a variant of the venerable paradox: "This sentence is false".

We see the same string appear twice, once with quotes (mention), and once
without (use), just like our Haskell quine. The unquoted part tempts the reader
to place it in quotes, akin to `show` in our Haskell quine. And Quine's
motivation for constructing his paradox was to avoid direct self-reference,
just as our programs must.

Knowing Quine's paradox might spoil the challenge for first-time quine authors,
which is why we only revealed it now.

== Combinator quines ==

Given a combinator term, suppose performing a single reduction counts as
running a program.
Define the $M$ is the combinator that Smullyan dubbed the "mockingbird" by:

\[
Mx = xx
\]

Then $MM$ is a quine, as $MM$ reduces to $MM$.

Even this tiny example features the above concepts. When reducing combinators,
the head combinator plays an active role (use) while its arguments play passive
roles (mention). Thus by printing one $M$ before another $M$, we have produced
a constructor-and-copier (the first $M$) and its quotation (the second $M$).

What if we say normalization is execution? (Recall this means that as
long as we're able to, we reduce the beginning of the expression.) Then the
problem is trivial: any combinator term in normal form is a quine.

Better is supplying a continuation or print syscall $f$ as the first argument
to a quine. We might say $x$ is a quine if:

\[
x f \rightarrow f x
\]

where an arrow denotes normalization. We imagine the syscall $f$
printing its argument $x$, which is the original program.
A term starting with $f$ is considered to be in weak normal form.

In this case, the $T$ combinator is a quine, since $Tf = fT$ for any $f$.

More interesting is the similar:

\[
x f \rightarrow f(x f)
\]

where we consider $xf$ to be complete original program: one could argue $x$
alone is incomplete because it needs to be given a syscall to work and $f$
ought to appear as well.

== Curry ==

A combinator $x$ such that for all $f$, both $x f$ and $f(x f)$ can reduce to
the same term is called a 'fixed-point combinator' (or the easier-to-say
'fixpoint combinator').

We use the equals sign to indicate this: $x f = f(x f)$. We've also been using
the equals sign to define combinators, so we rely on context to distinguish the
two. That is, in a combinator definition, the equals sign defines the meaning
of the combinator, otherwise it indicates that both sides can be reduced to the
same term. (Some authors avoid any possible confusion by writing, say, the
triple bar (&equiv;) for definitions.)

In mathematics, a fixed point of a function $f$ is a solution of $f(z) = z$,
and https://en.wikipedia.org/wiki/Fixed-point_theorems[fixed-point theorems are
important]. Thus $x$ is a called fixed-point combinator because $x f$ is a
fixed point of $f$ for any combinator $f$.

The most famous fixed-point combinator is the $Y$ combinator, commonly
attributed to Haskell Curry:

\[
Y = \lambda f . (\lambda x.f(x x)) (\lambda x. f (x x))
\]

However, the $Y$-combinator is unsuitable for our purposes: although both $Y f$
and $f(Y f)$ can be reduced to the same term, there is no way to reduce $Y f$
to $f(Y f)$.

Enter its less famous cousin, due to Alan Turing:

\[
\Theta = (\lambda x y.y(x x y))(\lambda x y.y(x x y))
\]

Then $\Theta$ is a quine:

\[
\Theta f \rightarrow f(\Theta f)
\]

The definitions of both $Y$ and $\Theta$ apply a combinator to itself,
reminding us of the $MM$ quine. This is no coincidence. We'll see fixed-point
combinators are a sort of parameterized version of $M$.

== Curry's Paradox ==

http://www.users.waitrose.com/~hindley/SomePapers_PDFs/2006CarHin,HistlamRp.pdf[Cardone
and Hindley, 'History of Lambda-calculus and Combinatory Logic'], page 8, state
that Turing's fixed-point combinator $\Theta$ was the first to be published
(1937, 'The p-function in &#0955;-K-conversion', in 'Journal of Symbolic Logic
2'), and the Y combinator was first published by Rosenbloom in 1950, in the
form:

\[
Y = \lambda x . W(Bx) (W(B x))
\]

where $W x y = x y y$.

However, in 1929 Curry wrote $(\lambda y.N(y y))(\lambda y.N(y y))$ in a letter
to Hilbert, where $N$ represents negation. It seems likely Curry had something
like the Y combinator in his head all along.

Another example is Curry's 1942 paper, 'The Inconsistency of Certain Formal
Logics', where he introduces what is now known as
https://en.wikipedia.org/wiki/Curry%27s_paradox[Curry's paradox]. Again, in
order to find a combinator that satisfies a certain recurrence, he writes down
$YX$ for a particular function $X$, rather than define $Y$ separately, before
employing a trick that we'll demonstrate shortly.

Thus we credit Curry for the Y combinator. This allows us to say Curry proved
Curry's paradox arises in certain logics with Curry's Y combinator!

What is Curry's paradox? Like Quine, Curry sought a paradox satisfying certain
constraints. Quine wanted to avoid direct self-reference, while Curry wanted to
avoid negation.

Russell's paradox is often told in the form of
https://en.wikipedia.org/wiki/Barber_paradox[a story about a barber]. It so
happens https://ncatlab.org/nlab/show/Curry%27s+paradox[Curry's paradox can be
viewed as a generalization of Russell's paradox] which changes the barber
paradox to the following.

Let $P$ be any proposition, such as "Germany borders China". The more ludicrous
the better.

Curry's barber shaves all men who, if they shaved themselves, then $P$ is true.
Does the barber shave himself?

If so, then $P$ is true, because the barber shaves men for which this strange
implication holds.

We just said if the barber shaves himself, then $P$ is true. Thus the barber is
exactly the kind of man the barber shaves. So yes, the barber does shave
himself. This implies $P$. That is, any proposition $P$ is true!

== Fixed-point Combinators ==

How were fixed-point combinators found? Let's try to force our way through, and
simply define:

\[
X f = f(X f)
\]

This definition of $X$ is illegal because $X$ also appears on the right-hand
side. Again, we encounter the problem of a self-description attempting to
contain itself.

We can make it legal by replacing $X$ on the right-hand side a new variable
$x$, which we'll introduce on the left-hand side:

\[
X' x f = f(x f)
\]

It may seem unclear how this helps, but one weird trick is all we need. If we
apply all occurences of `x` on the right-hand side to themselves, we are in a
position to exploit the use-mention distinction:

\[
X'' x f = f(x x f)
\]

We find:

\[
X'' X'' f \rightarrow f(X'' X'' f)
\]

Thus $X = X''X''$ solves our equation above. The first $X''$ is a use, the
second a mention. As a lambda term, we have:

\[
X'' = \lambda x f . f (x x f)
\]

So we have in fact just derived Turing's $\Theta$ combinator.

Here's an alternate solution. We adopt Curry's point of view and treat $f$
as a given function; a free variable.

We try to use brute force to define a fixed point of $f$:

\[
Z = f Z
\]

We apply the use-mention trick to remove the $Z$ on the right-hand side:

\[
\begin{align}
Z' z &= f z z \\
Z &= Z' Z' = (\lambda z . f z z)(\lambda z . f z z)
\end{align}
\]

If we now bind $f$ with a lambda, we wind up with the $Y$ combinator:

\[
Y = \lambda f . (\lambda z . f z z)(\lambda z . f z z)
\]

== Recursion ==

Recursion is intimately connected to quines. A recursive function can be
considered a self-replicating function, except instead of printing its clone,
it calls it with different parameters.

We typically define a recursive function $F$ with equations where $F$ also
appears on the right-hand side, that is, the self-description contains a
description of itself. For instance:

------------------------------------------------------------------------
fib n = add (fib (pred n)) (fib (pred (pred n)))
------------------------------------------------------------------------

As above, such a definition is illegal in the world of combinators:

\[
F = \lambda n . A (F (P n)) (F (P (P n)))
\]

And as above, we can fix this with the use-mention trick:

\[
\begin{align}
G &= \lambda f n . A (f f (P n)) (f f (P (P n))) \\
F &= G G
\end{align}
\]

However, now that we have derived $Y$ and $\Theta$, a simpler approach is to
replace each $F$ with a single copy of a variable bound in a new outermost
lambda:

\[
G = \lambda f n . A (f (P n)) (f (P (P n))))
\]

and pass the whole thing to a fixed-point combinator like $\Theta$:

\[
F = \Theta G
\]

Then:

\[
\begin{align}
F & \rightarrow \Theta G \\
& \rightarrow G(\Theta G) \\
& = G(F) \\
& \rightarrow \lambda f n . A (f (P n)) (f (P (P n)))) F \\
& \rightarrow \lambda n . A (F (P n)) (F (P (P n))))
\end{align}
\]

Using a predefined fixed-point combinator for recursion may be preferable to
always applying the use-mention self-application trick because:

  * The lambda terms are simpler.
  * When translated, the combinator terms are simpler.
  * In some languages, recursive functions fail to type-check, while the
  argument of a fixed-point combinator may be legal. Adding a built-in
  fixed-point combinator allows us to run it as intended, while clearly
  demarcating the only parts of the program that might loop forever.

We can explore fixed-point combinators in Haskell:

------------------------------------------------------------------------
import Data.Function (fix)
fib = fix (\f n -> if n <= 1 then n else (+) (f (pred n)) (f (pred (pred n))))
testFib20 = fib 20 == 6765
------------------------------------------------------------------------

== Von Neumann Paradox ==

There is also https://en.wikipedia.org/wiki/Von_Neumann_paradox[a paradox
bearing von Neumann's name], and since we already discussed Quine's and Curry's
paradoxes, it seems only fair to bring it up. Roughly speaking, among other
things, the von Neumann paradox uses the axiom of choice to show how to cut up
a unit square and shuffle around the pieces to form two unit squares.

In logics based on lambda calculus or combinators, we can stay consistent by
introducing types to disallow pathogens like fixed-point combinators which lead
to paradoxes.

In mathematics, paradoxes like von Neumann's are strange but tolerable. While
one can live without the axiom of choice, most prefer to keep it.

== A Bonus Paradox ==

In Haskell, the type of the fixed-point combinator is `(a -> a) -> a`.
Transporting this to logic via the Curry-Howard correspondence, this is
"for all A, if A implies A, then A is true". Since any proposition implies
itself, this implies any proposition is true.

== Epilogue ==

Some authors shorten the definitions of $Y$ and $\Theta$ by using
$M = \lambda x.x x$ to apply a term to itself.

We could have defined:

\[
\begin{align}
Y &= BM(RMB) \\
\Theta &= M(B(SI)M) \\
M &= SII
\end{align}
\]

Our remarks mostly apply if we choose weak head normalization instead of
normalization, though $M = SII$ no longer works.

Smullyan, in _Diagonalization and Self-Reference_, attempts natural-language
versions of some of the above, but
https://www.jamesrmeyer.com/paradoxes/smullyan-paradox.html[his examples turn
out to be flawed].

Amazingly, von Neumann figured out self-replication years before DNA was
discovered. DNA can be viewed as a program. In some contexts, DNA plays a
passive role (mention) and is simply duplicated, and in others, DNA plays an
active role (use) and causes various amino acids to combine in certain ways
to build things.

Turing is well-known for helping crack link:../haskell/enigma.html[the Enigma
Machine] in World War II. Perhaps less known is that Quine was also a key
figure in Allied military intelligence. He served in the United States Navy,
deciphering messages from German submarines.
