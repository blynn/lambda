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
<div id="surpriseP" hidden>; The Surprise from "The Roots of Lisp" by Paul Graham.
; The "eval." function evaluates a given Lisp expression.
; We demonstrate it by running it on the `subst` example in the paper.

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
</div>
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

A single file will serve as the source for:

  - the text in this webpage
  - the JavaScript for this webpage
  - a command-line Lisp interpreter

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
nodes on a full binary tree. The `atom` primitive of Lisp has a slightly
confusing in that it returns true for all atoms and the empty list; presumably
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
  show (l :^ r)    = "(" ++ show l ++ " " ++ showR r ++ ")" where
    showR (l :^ Lf "") = show l
    showR (l :^ r)     = show l ++ " " ++ showR r
    showR x            = show x
\end{code}

Haskell is more deserving of the term "list processor" because its lists
are true singly-linked lists. This may not always be good:
https://www.schoolofhaskell.com/user/bss/magma-tree[lists are less general than
trees because monoids are fussier than magmas].

While we can easily define other data structures, they will never be as
easy to use as lists in Haskell. Aside from the extensive library support,
notation such as `[1, 2, 3]` and `(h:t)` only applies to lists.

Infix constructors, the FTP proposal, and view patterns have improved matters,
but lists are still king. Indeed, initially I wrote the interpreter with:

------------------------------------------------------------------------------
data Expr = Atom String | List [Expr]
------------------------------------------------------------------------------

so I could write:

------------------------------------------------------------------------------
  f "quote" [x] = x
  f "atom" [Atom _ ] = Atom "t"
  f "atom" [List []] = Atom "t"
  f "atom" [_      ] = List []
  f "atom" [Atom _ ] = Atom ""
  f "car"  [List (h:_)] = h
  f "cdr"  [List (_:t)] = List t
  f "cons" [h,  List t] = List (h:t)
  ...
------------------------------------------------------------------------------

This felt like cheating, so ultimately I opted to define tree structure from
scratch.

== 7 Primitive Operators ==

To take advantage of Haskell's pattern matching we introduce a function `f`
to handle primitive operators. Some details will be handled later.
For example, for `quote` and `cond`, we must avoid evaluating any arguments,
while all other primitives require all their arguments to be evaluated first.

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
environment.

For `lambda`, we modify the environment accordingly before recursing.
Otherwise, it's clear when we should look up a binding in the environment, or
evaluate other nodes of the tree before calling `f`.

Again we see the overheads incurred by using a non-list data structure in
Haskell. In my initial list-based code, I could simply use the default `map`
instead of  `mapL`, and `fromTree` would be superfluous.

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
main = withElems ["input", "output", "evalB", "surpriseB", "surpriseP"] $
  \[iEl, oEl, evalB, surB, surP] -> do
  surB  `onEvent` Click $ const $ do
    getProp surP "innerHTML" >>= setProp iEl "value"
    setProp oEl "value" ""

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
