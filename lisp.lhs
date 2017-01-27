= Lisp in Haskell =

Haskell is Lisp with modern conveniences. Haskell is a fashionable five-star
high-tech luxurious language, but
http://newartisans.com/2009/03/hello-haskell-goodbye-lisp/[stripping away its
contemporary furnishings reveals a humble core surprisingly similar to Lisp].

This is because
http://research.microsoft.com/en-us/um/people/simonpj/papers/history-of-haskell/[functional programming languages began with John McCarthy's invention of Lisp].
Thus to study the roots of Haskell, we should study the roots of Lisp.
http://www-formal.stanford.edu/jmc/recursive.html[McCarthy's classic 1960 paper]
remains an excellent source, but Paul Graham's homage
http://www.paulgraham.com/rootsoflisp.html['The Roots of Lisp']
[http://languagelog.ldc.upenn.edu/myl/ldc/llog/jmc.pdf[PDF]] is far more
accessible, and corrects bugs in the original paper.

We'll build a Lisp interpreter based on Graham's paper:

[pass]
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
<script src="lisp.js"></script>
<p><button id="evalB">Run</button>
<button id="surpriseB">Surprise Me!</button>
<button id="quoteB">Quote Quiz</button>
</p>
<p><textarea style="border: solid 2px; border-color: #999999" id="input" rows="10" cols="80">(defun subst (x y z)  ; From "The Roots of Lisp" by Paul Graham.

  (cond ((atom z)
         (cond ((eq z y) x)
               ('t z)))
        ('t (cons (subst x y (car z))
                  (subst x y (cdr z ))))))

(subst 'm 'b '(a b (a b c) d))  ; We expect (a m (a m c) d).
</textarea></p>
<p><textarea id="output" rows="3" cols="80" readonly></textarea></p>
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

To build everything yourself, install http://haste-lang.org/[Haste] and
http://asciidoc.org[AsciiDoc], and then type:

------------------------------------------------------------------------------
$ haste-cabal install parsec
$ wget https://crypto.stanford.edu/~blynn/haskell/lisp.lhs
$ hastec lisp.lhs
$ sed 's/^\\.*{code}$/-----/' lisp.lhs | asciidoc -o - - > lisp.html
$ cabal install parsec readline
$ ghc lisp.lhs
------------------------------------------------------------------------------

Open `lisp.html` in a browser, or run `./lisp`.

A single source file forces us to tolerate the presence of conditional
compilation macros. For JavaScript, we need a few
http://haste-lang.org/[Haste] imports.

For our command-line interpreter, the GNU Readline library is a boon.
Also, if it appears that an expression is incomplete, rather than complain
immediately, our interpreter asks for another line. This feature requires
importing the Parsec library's Error module.

\begin{code}
{-# LANGUAGE CPP #-}
#ifdef __HASTE__
import Haste.DOM
import Haste.Events
#else
import System.Console.Readline
import Text.ParserCombinators.Parsec.Error
#endif
\end{code}

The interpreter itself only needs a few imports:

\begin{code}
import Control.Monad
import Data.List
import Text.ParserCombinators.Parsec
\end{code}

== Tree Processor ==

We define a data structure that will hold a Lisp expression. Lisp should
really be called Trep: it processes trees, not lists!

Atoms and the empty list are external nodes and non-empty lists are internal
nodes on a full binary tree. The `atom` primitive of Lisp is slightly
confusing because it returns true for all atoms and the empty list; presumably
the name `is-external-node` is too unwieldy.

To emphasize that Lisp expressions are binary trees, we construct external node
with `Lf`, which stands for "leaf", and use the infix constructor `(:^)` to
build internal nodes from left and right subtrees.

The infix constructor also makes pattern matching easier. We give it
right-associativity so we can write Lisp lists with fewer parentheses, because
a flattened Lisp list is actually an unbalanced binary tree whose spine extends
to the right and terminates with a sentinel node (`nil`), and the elements of
the list are the left child nodes.

Our interpreter accepts a sequence of expressions, and later expressions may
use labels defined by earlier expressions. To help with this, we add the
`Label` constructor. This design is unclean, as it means a `label` in the wrong
place will cause problems, but it suffices for our demo.

We also define `Bad` for crude but serviceable error-handling.

\begin{code}
infixr 5 :^
data Expr = Lf String | Expr :^ Expr | Label String Expr | Bad

instance Show Expr where
  show Bad         = "?"
  show (Label s _) = s
  show (Lf "")     = "()"
  show (Lf s)      = s
  show (l :^ r)    = "(" ++ show l ++ showR r ++ ")" where
    showR (Lf "")  = ""
    showR (l :^ r) = " " ++ show l ++ showR r
    showR x        = " " ++ show x
\end{code}

Haskell is more deserving of the term "list processor" because its lists
are true singly-linked lists, and the language encourages their use thanks to
extensive library support and notation such as `[1, 2, 3]` and `(h:t)`.
This may not always be good:
https://www.schoolofhaskell.com/user/bss/magma-tree[lists are less general than
trees because monoids are fussier than magmas].

While we can easily define other data structures, they will never be as easy to
use. Infix constructors, the FTP proposal, and view patterns have improved
matters, but lists are still king. Indeed, initially I wrote the interpreter
with:

------------------------------------------------------------------------------
data Expr = Atom String | List [Expr]
------------------------------------------------------------------------------

so I could write:

------------------------------------------------------------------------------
  f "quote" [x] = x
  f "atom" [Atom _ ] = Atom "t"
  f "atom" [List []] = Atom "t"
  f "atom" [_      ] = List []
  f "car"  [List (h:_)] = h
  f "cdr"  [List (_:t)] = List t
  f "cons" [h,  List t] = List (h:t)
  ...
------------------------------------------------------------------------------

This felt like cheating, so ultimately I defined a tree structure from scratch.

== 7 Primitive Operators ==

To take advantage of Haskell's pattern matching we introduce a function `f`
to handle primitive operators. We write this function first, and worry about
how it fits into the bigger picture later.

\begin{code}
eval env = g where
  f "quote" (x :^ Lf "") = x

  f "atom" (Lf _ :^ Lf "") = Lf "t"
  f "atom" _               = Lf ""

  f "eq" (Lf x :^ Lf y :^ Lf "") | x == y    = Lf "t"
                                 | otherwise = Lf ""
  f "eq" (_    :^ _    :^ Lf "") = Lf ""

  f "car"  ((l :^ r) :^ Lf "") = l
  f "cdr"  ((l :^ r) :^ Lf "") = r
  f "cons" ( l :^ r  :^ Lf "") = l :^ r

  f "cond" ((l :^ r :^ Lf "") :^ rest) | Lf "t" <- g l = g r
                                       | otherwise     = f "cond" rest
\end{code}

I could have added a `Nil` sentinel to the `Expr` data type and used it
instead of `Lf ""`, but the code seemed more honest without it.

The `f` function is also a good place to handle `label`, and some Lisp
shorthand:

\begin{code}
  f "label" (Lf s :^ e :^ Lf "") = Label s e

  f "defun" (s :^ etc) = g $ Lf "label" :^ s :^ (Lf "lambda" :^ etc) :^ Lf ""
  f "list" x = x

  f _ _ = Bad
\end{code}

== Eval ==

Locally, we give the name `g` to the evaluation function for a given
environment:

 * For labels, we add a binding to the environment, then evaluate the newly
bound function on the given arguments.

 * For atoms, we look up its binding in the environment.

 * For `lambda`, we modify the environment accordingly before recursing.

 * Otherwise we expect the left child of the root to be a leaf. If it has a
binding in the environment, we recursively evaluate it on the rest of the tree.
(This means it's possible to override the builtin functions.) If it's `cond`,
`quote`, `defun`, or `label`, then we call `f` without evaluating its arguments
first. If not, then we do evaluate the arguments first before calling `f`.

Again we see the overheads incurred by using a non-list data structure in
Haskell. In my initial list-based code, I could simply use the default `map`
instead of  `mapL`, and `fromTree` would be unneeded.

\begin{code}
  g (Label s e :^ r) = eval ((s, e):env) $ e :^ r
  g (Lf s) | Just b <- lookup s env = b
           | otherwise              = Bad

  g ((Lf "lambda" :^ args :^ body :^ Lf "") :^ t) =
    eval (zip (fromLeaf <$> fromTree args) (fromTree $ mapL g t) ++ env) body
  g (Lf h :^ t)
    | Just b <- lookup h env                  = g $ b :^ t
    | elem h $ words "cond quote defun label" = f h t
    | otherwise                               = f h $ mapL g t
  g _ = Bad

  fromTree (Lf "")  = []
  fromTree (h :^ t) = h:fromTree t
  fromLeaf (Lf x)   = x

  mapL f (Lf "")  = Lf ""
  mapL f a@(Lf _) = f a
  mapL f (l :^ r) = f l :^ mapL f r
\end{code}

== Parser and User Interface ==

A simple parser suits Lisp's simple grammar:

\begin{code}
expr :: Parser Expr
expr = between ws ws $ atom <|> list <|> quot where
  ws   = many $ void space <|> comm
  comm = void $ char ';' >> manyTill anyChar (void (char '\n') <|> eof)
  atom = Lf <$> many1 (alphaNum <|> char '.')
  list = foldr (:^) (Lf "") <$> between (char '(') (char ')') (many expr)
  quot = char '\'' >> expr >>= pure . (Lf "quote" :^) . (:^ Lf "")

oneExpr = expr >>= (eof >>) . pure
\end{code}

We use a list to store bindings. We permanently augment the global environment
when the `eval` function returns a `Label`.

We preload definitions of the
form `(defun cadr (x) (cdr (car x)))` from `caar` to `cddddr`.

\begin{code}
addEnv (Label s e) = ((s, e):)
addEnv _           = id

preload = foldl' f [] $ concat $ genCadr <$> [2..4] where
  f env s = let Right expr = parse oneExpr "" s in addEnv (eval env expr) env

genCadr n = [concat ["(defun c", s, "r (x) (c", [h], "r (c", t, "r x)))"] |
  s@(h:t) <- replicateM n "ad"]
\end{code}

For this webpage, we set up one button to run the program supplied in the input
textarea and place the results in the output textarea. Another button fills
the input textarea with a predefined program.

\begin{code}
#ifdef __HASTE__
main = withElems ["input", "output", "evalB",
                  "surpriseB", "surpriseP",
                  "quoteB", "quoteP"] $
  \[iEl, oEl, evalB, surB, surP, quoB, quoP] -> do
  surB  `onEvent` Click $ const $ do
    getProp surP "innerHTML" >>= setProp iEl "value" >> setProp oEl "value" ""
  quoB  `onEvent` Click $ const $ do
    getProp quoP "innerHTML" >>= setProp iEl "value" >> setProp oEl "value" ""

  evalB `onEvent` Click $ const $ do
    let
      run (out, env) expr = case r of
        Label _ _ -> (out, env1)
        r         -> (out ++ show r ++ "\n", env1)
        where r    = eval env expr
              env1 = addEnv r env
    s <- getProp iEl "value"
    setProp oEl "value" $ case parse (many expr >>= (eof >>) . pure) "" s of
      Left  e  -> "Error: " ++ show e ++ "\n"
      Right es -> fst $ foldl' run ("", preload) es
#else
\end{code}

[pass]
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
<textarea id="surpriseP" hidden>; The Surprise from "The Roots of Lisp" by Paul Graham.
; The "eval." function evaluates a given Lisp expression.
; We demonstrate it on the `subst` example in the paper.

(defun null. (x)
  (eq x '()))

(defun and. (x y)
  (cond (x (cond (y 't) ('t '())))
        ('t '())))

(defun not. (x)
  (cond (x '())
        ('t 't)))

(defun append. (x y)
  (cond ((null. x) y)
        ('t (cons (car x) (append. (cdr x) y)))))

(defun pair. (x y)
  (cond ((and. (null. x) (null. y)) '())
        ((and. (not. (atom x)) (not. (atom y)))
         (cons (list (car x) (car y))
               (pair. (cdr x) (cdr y))))))

(defun assoc. (x y)
  (cond ((eq (caar y) x) (cadar y))
        ('t (assoc. x (cdr y)))))

(defun eval. (e a)
  (cond
    ((atom e) (assoc. e a))
    ((atom (car e))
     (cond
       ((eq (car e) 'quote) (cadr e))
       ((eq (car e) 'atom)  (atom   (eval. (cadr e) a)))
       ((eq (car e) 'eq)    (eq     (eval. (cadr e) a)
                                    (eval. (caddr e) a)))
       ((eq (car e) 'car)   (car    (eval. (cadr e) a)))
       ((eq (car e) 'cdr)   (cdr    (eval. (cadr e) a)))
       ((eq (car e) 'cons)  (cons   (eval. (cadr e) a)
                                    (eval. (caddr e) a)))
       ((eq (car e) 'cond)  (evcon. (cdr e) a))
       ('t (eval. (cons (assoc. (car e) a)
                        (cdr e))
                  a))))
    ((eq (caar e) 'label)
     (eval. (cons (caddar e) (cdr e))
            (cons (list (cadar e) (car e)) a)))
    ((eq (caar e) 'lambda)
     (eval. (caddar e)
            (append. (pair. (cadar e) (evlis. (cdr e) a))
                     a)))))

(defun evcon. (c a)
  (cond ((eval. (caar c) a)
         (eval. (cadar c) a))
        ('t (evcon. (cdr c) a))))

(defun evlis. (m a)
  (cond ((null. m) '())
        ('t (cons (eval. (car m) a)
                  (evlis. (cdr m) a)))))

; We expect (a m (a m c) d).
(eval. '(
  (label subst (lambda (x y z)
    (cond ((atom z)
           (cond ((eq z y) x)
                 ('t z)))
          ('t (cons (subst x y (car z))
                    (subst x y (cdr z )))))))
  'm 'b '(a b (a b c) d)) '())
</textarea>
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The command-line variant of our program provides more immediate gratification,
printing results on every newline if it terminates a valid expression.

\begin{code}
expectParen (Expect "\"(\"") = True
expectParen (Expect "\")\"") = True
expectParen _                = False

repl pre env = do
  ms <- readline $ if null pre then "> " else ""
  case ms of
    Nothing -> putStrLn ""
    Just s  -> addHistory s >> case parse oneExpr "" $ pre ++ s of
      Left err  -> case find expectParen $ errorMessages err of
        Nothing -> do
          putStrLn $ "parse error: " ++ show err
          repl "" env
        _ -> repl (pre ++ s ++ "\n") env
      Right expr -> do
        let r = eval env expr
        print r
        repl "" $ addEnv r env

main = repl "" preload
#endif
\end{code}

And with that, our interpreter is done!

== Family Resemblance ==

The most obvious difference between Haskell and Lisp is the syntax, but
this is primarily syntax sugar. For example, take a typical
function from Paul Graham's 'On Lisp':

------------------------------------------------------------------------------
(defun our-remove-if (fn lst)
  (if (null lst)
    nil
    (if (funcall fn (car lst))
      (our-remove-if fn (cdr lst))
      (cons (car lst) (our-remove-if fn (cdr lst))))))
------------------------------------------------------------------------------

We could write it almost identically in Haskell:

------------------------------------------------------------------------------
if' a b c = if a then b else c

ourRemoveIf fn lst =
  (if' (null lst)
    []
    (if' (fn (head lst))
      (ourRemoveIf fn (tail lst))
      ((:) (head lst) (ourRemoveIf fn (tail lst)))))
------------------------------------------------------------------------------

but it's better to add generous servings of Haskell syntax sugar.

------------------------------------------------------------------------------
ourRemoveIf _ []                 = []
ourRemoveIf f (x:xs) | f x       = ourRemoveIf f xs
                     | otherwise = x : ourRemoveIf f xs
------------------------------------------------------------------------------

In this small example we can see various sweeteners: meaningful indentation;
pattern matching; guards; infix and prefix notation; concise notation for
lists.

There is in fact some substance behind the delicious style. Patterns are
sophisticated enough to be useful, yet elementary enough so compilers can
detect overlapping patterns or incomplete patterns in a function definition.
This catches bugs that would go unnoticed in a `cond` (or `case` in Haskell).

With this in mind, we see the source of our interpreter is almost the same as
The Surprise, except it's less cluttered and more robust. For example, for
the 7 primitives, thanks to pattern matching, the function `f` reduces
duplicated code such as `eq (car e)` gives the compiler more power, and detects
errors when the wrong number of arguments are supplied.

By the way, as with Lisp, in reality we would never bother defining the above
function, because `ourRemoveIf = filter . (not .)`.

== Less is more ==

Haskell is really the Core language, coated in heavy layers of syntax sugar.
The
https://ghc.haskell.org/trac/ghc/browser/ghc/compiler/coreSyn/CoreSyn.hs[Core
grammar] only takes a handful of lines:

------------------------------------------------------------------------------
data Expr b
  = Var   Id
  | Lit   Literal
  | App   (Expr b) (Arg b)
  | Lam   b (Expr b)
  | Let   (Bind b) (Expr b)
  | Case  (Expr b) b Type [Alt b]
  | Cast  (Expr b) Coercion
  | Tick  (Tickish Id) (Expr b)
  | Type  Type
  | Coercion Coercion
  deriving Data

type Arg b = Expr b

data Bind b = NonRec b (Expr b)
            | Rec [(b, (Expr b))]
------------------------------------------------------------------------------

Parallels with the Lisp are obvious, for example, `Lam` is `lambda`, `Case` is
`cond`, and `App` is the first cons cell in a Lisp list, There's bookkeeping
for types, and source annotation (`Tick`) for profilers and similar tools, but
otherwise Core and Lisp share the same minmalist design.

== Core changes ==

Nonetheless, there are profound differences between Haskell and Lisp. Although
almost invisible to the untrained eye, they have enormous implications.

Haskell has benefited from the experience of functional programmers, as well as
advances in theory involving the foundations of mathematics and computer
science.

 - Lisp programmers pioneered programming with pure functions. Haskell has
   taken purity to heart: its type system means the compiler knows which
   functions are pure and which are not, and steers programmers to isolate
   impure code in tiny functions, leaving the rest of the code pure.

 - The discovery that the
   https://en.wikipedia.org/wiki/Monad_(category_theory)[monads of category
   theory] applied to programming meant Haskell could stay pure yet handle I/O
   beautifully.

 - Lazy evaluation largely obviates the need for macros. Good language support
   for pure functions is a prerequisite for effective lazy evaluation.

 - The
   https://en.wikipedia.org/wiki/Hindley%E2%80%93Milner_type_system[Hindley-Milner
   type system] underpinning Haskell 98 lets us write code without a single
   type annotation, so it still feels like Lisp, yet an efficient type
   inference algorithm means the compiler rejects badly typed programs. Haskell
   has since gone beyond Hindley-Milner, but even so, type annotation is
   inconspicuous.

 - The Core language is built on https://en.wikipedia.org/wiki/System_F[System
   F], which is formalizes parametric polymorphism and also guarantees programs
   terminate. This guarantee is voided by Haskell's support for recursion, but
   we can regain it by restricting recursion, as
   http://www.idris-lang.org/[Idris] does.

 - https://ghc.haskell.org/trac/ghc/wiki/Commentary/Compiler/FC[Core pushes
   beyond System F to enrich types], but not so much more that we need
   full-blown https://en.wikipedia.org/wiki/Dependent_type[dependent types].
   (This explains the weirder parts of the Core grammar.)

 - The https://en.wikipedia.org/wiki/Curry%E2%80%93Howard_correspondence[Curry-Howard correspondence] and Haskell's expressive type system means more of the program can be shown to be correct at compile-time.

 - Roughly speaking, Lisp reads `(min 1 x)` as `(min (1 x))`, while Haskell
   reads it as `((min 1) x)`. For function evaluation, Haskell's parse tree is
   more troublesome because we must repeatedly traverse left from the root
   until we find the function to evaluate, rather than simply take the left
   child of the root. However, it's a net win because a curried function is
   simply a subtree. We have
   https://en.wikipedia.org/wiki/Combinatory_logic[combinatory logic] to thank
   for left-associative function application.

 - Lisp is perhaps the best language for appreciating the equivalence of code
   and data, since a program is its own representation. However, although
   artistically and intellectually engaging, this muddying of the use-mention
   waters trips up everyone from students
   (https://www.cs.kent.ac.uk/people/staff/dat/miranda/wadler87.pdf[who have
   trouble with `quote`]) and theorists
   (http://www.cs.bc.edu/~muller/research/papers.html#toplas[who have trouble
   formally reasoning about it]). Haskell wisely chose
   http://research.microsoft.com/en-us/um/people/simonpj/papers/meta-haskell/[a
   more explicit form of metaprogramming].

[pass]
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
<textarea id="quoteP" hidden>
; Students are confused by these exercises by Abelson and Sussman.
; Philip Wadler, "Why calculating is better than scheming".

; What are the values of the following expressions?

(car ''abracadabra)
(cdddr '(this list contains '(a quote)))
(car (quote (a b)))
</textarea>
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

 - Lisp was inspired by link:lambda.html['lambda calculus']: we've seen the
   language is mostly `lambda` plus seven primitives. Scheme inches a little
   closer. Haskell has the courage of its convictions to go (almost) all the
   way: Haskell is a typed lambda calculus (plus let statements, allowing
   unrestricted recursion).

 - McCarthy's `eval` function must have been astonishing for its time.
   Graham calls it The Surprise. By adding a handful of primitives to lambda
   calculus, we can write a self-interpreter that fits on a page. Try doing
   that with Turing machines! But it turns with plain unadulterated lambda
   calculus, we can write
   http://repository.readscheme.org/ftp/papers/topps/D-128.pdf[a
   self-interpreter that fits on one line]: +
------------------------------------------------------------------------------
(λf.(λx.f(xx))(λx.f(xx)))(λem.m(λx.x)(λmn.em(en))(λmv.e(mv)))
------------------------------------------------------------------------------

A casual observer might accuse Lisp of being too expedient. But criticism is
easy with hindsight. Would Haskell ever been designed without the lessons
learned from Lisp? Without Lisp, would we still be stuck with Turing machines
for the theory of computation? Would language researchers have bothered to
rummage through logic and category theory?
