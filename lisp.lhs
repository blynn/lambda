= Lisp in Haskell =

http://research.microsoft.com/en-us/um/people/simonpj/papers/history-of-haskell/[Functional programming languages began with John McCarthy's invention of Lisp].

Haskell is Lisp with modern conveniences. Haskell is a fashionable five-star
high-tech luxurious language, but stripping away its contemporary furnishings
reveals a humble core surprisingly similar to Lisp.

Thus to study the roots of Haskell, we should study the roots of Lisp.
http://www-formal.stanford.edu/jmc/recursive.html[McCarthy's classic 1960 paper]
remains an excellent source, but Paul Graham's homage
http://www.paulgraham.com/rootsoflisp.html['The Roots of Lisp']
[http://languagelog.ldc.upenn.edu/myl/ldc/llog/jmc.pdf[PDF]] is far more
accessible, and corrects bugs in the original paper.

We'll build an interpreter that can run Graham's Lisp code:

[pass]
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
<script src="lisp.js"></script>
<p><button id="evalB">Run</button>
<button id="surpriseB">Surprise Me!</button>
</p>
<p><textarea style="border: solid 2px; border-color: #999999" id="input" rows="10" cols="80"></textarea></p>
<p><textarea id="output" rows="3" cols="80" readonly></textarea></p>
<div id="surpriseP" hidden>; The Surprise. See Paul Graham, "The Roots of Lisp".
; The "eval." function takes a Lisp expression and evaluates it.
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

; We expect "(a m (a m c) d)".
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

Open `lisp.html` in a browser, or run "./lisp".

A single source file forces us to tolerate the presence of conditional
compilation macros. For JavaScript, we need a few
http://haste-lang.org/[Haste] imports:

\begin{code}
{-# LANGUAGE CPP #-}
#ifdef __HASTE__
import Haste.DOM
import Haste.Events
\end{code}

For our command-line interpreter, the GNU Readline makes life much easier.
Also, if it appears that an expression is incomplete, rather than complain
immediately, our interpreter will ask for another line. These require certain
imports:

\begin{code}
#else
import System.Console.Readline
import Text.ParserCombinators.Parsec.Error
#endif
\end{code}

Onwards to the common code.

\begin{code}
import Control.Monad
import Data.List
import Text.ParserCombinators.Parsec

data Expr = Atom String | List [Expr] | Label String Expr | Bad

instance Show Expr where
  show (Label s _) = s
  show (Atom s)    = s
  show (List as)   = "(" ++ unwords (show <$> as) ++ ")"
  show Bad         = "?"

eval env = g where
  f "quote" [x] = x

  f "atom" [Atom _ ] = Atom "t"
  f "atom" [List []] = Atom "t"
  f "atom" [_      ] = List []

  f "eq" [List [], List []] = Atom "t"
  f "eq" [Atom x , Atom y ] | x == y    = Atom "t"
                            | otherwise = List []
  f "eq" [_      , _      ] = List []

  f "car"  [List (h:_)] = h
  f "cdr"  [List (_:t)] = List t
  f "cons" [h,  List t] = List (h:t)

  f "cond" []                = List []
  f "cond" (List [p, e]:rest) = case g p of Atom "t" -> g e
                                            _        -> f "cond" rest

  f "label" [Atom id, e] = Label id e

  f "defun" [id, ps, e] =
    g $ List [Atom "label", id, List [Atom "lambda", ps, e]]

  f "list" t = List t

  f _ _ = Bad

  -- Convenient, but we can live without these.
  -- g t@(Atom "t") = t
  -- g empty@(List []) = empty

  g (List (Label id e:rest)) = eval ((id, e):env) $ List $ e:rest
  g (Atom s) | Just b <- lookup s env = b
             | otherwise              = Bad
  g (List (List [Atom "lambda", List args, body]:t))
    = eval (zip (fromAtom <$> args) (g <$> t) ++ env) body where
     fromAtom (Atom p) = p
  g (List (List h:t)) = g $ List $ g (List h):t
  g (List (Atom h:t))
    | Just b <- lookup h env                  = g $ List $ b:t
    | elem h $ words "cond quote defun label" = f h t
    | otherwise                               = f h $ g <$> t
  g _ = Bad

expr :: Parser Expr
expr = between ws ws $ atom <|> list <|> quote where
  ws = many $ void space <|> comm
  comm = void $ char ';' >> manyTill anyChar ((void $ char '\n') <|> eof)
  atom = Atom <$> many1 (alphaNum <|> char '.')
  list = List <$> between (char '(') (char ')') (many expr)
  quote = char '\'' >> expr >>= pure . List . (Atom "quote":) . pure
  qquote = do
    char '\''
    x <- expr
    return $ List [Atom "quote", x]

oneExpr = expr >>= (eof >>) . pure

addEnv (Label s e) = ((s, e):)
addEnv _           = id

-- Preload definitions such as "(defun cadr (x) (cdr (car x)))".
preload = foldl' f [] $ concat $ genCadr <$> [2..4] where
  f env s = let Right expr = parse oneExpr "" s in addEnv (eval env expr) env

genCadr n = [concat ["(defun c", s, "r (x) (c", [h], "r (c", t, "r x)))"] |
  s@(h:t) <- replicateM n "ad"]

#ifdef __HASTE__
main = withElems ["input", "output", "evalB", "surpriseB", "surpriseP"] $ \[iEl, oEl, evalB, surB, surP] -> do
  surB  `onEvent` Click $ const $
    getProp surP "innerHTML" >>= setProp iEl "value"

  evalB `onEvent` Click $ const $ do
    let
      run (out, env) expr = case r of
        Label _ _ -> (out, env1)
        r         -> (out ++ show r ++ "\n", env1)
        where
          r = eval env expr
          env1 = addEnv r env
    s <- getProp iEl "value"
    setProp oEl "value" $ case parse (many expr >>= (eof >>) . pure) "" s of
      Left e -> "Error: " ++ show e ++ "\n"
      Right es -> fst $ foldl' run ("", preload) es
#else
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
