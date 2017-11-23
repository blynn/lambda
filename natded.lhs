= Deduced again (naturally) =

When mathematicians first studied formal logic, they built
https://en.wikipedia.org/wiki/Hilbert_system[Hilbert systems], where proving
theorems corresponds to programming with link:sk.html[combinators].

APL and its ilk show the power of this approach. However, some functions are
easier to describe if we can name their arguments. In logic, analogous concerns
drove Gentzen to devise
'https://en.wikipedia.org/wiki/Natural_deduction[natural deduction]':

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

[pass]
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
<script src="natded.js"></script>
<p>
<span id="preT"></span>
</p>
<div id="ruleBar" style="display:none;">
<div id="hypoDiv" style="display:none;">
<p>
<input id="hypoT" cols="64" type="text">
<button id="newHypoB">New Hypothesis</button>
<div><span id="errT"></span></div>
</p>
</div>
<p>
<style>.logic{cursor:pointer;border:2px solid blue;border-radius:10px;padding:5px;margin:5px;}</style>
<!-- I wanted &rArr;&#120024; and &rArr;&#120020;
but some browsers lack the fonts to display these. -->
<span class="logic" id="impliesI">&rArr;I</span>
<span class="logic" id="impliesE">&rArr;E</span>
<span class="logic" id="andI">&and;I</span>
<span class="logic" id="and1E">&and;1E</span>
<span class="logic" id="and2E">&and;2E</span>
<span class="logic" id="or1I">&or;1I</span>
<span class="logic" id="or2I">&or;2I</span>
<span class="logic" id="orE">&or;E</span>
<span class="logic" id="falseE">&perp;E</span>
<span class="logic" id="notNot">LEM</span>
</p>
<p style="float:right;">
<span class="logic" id="undoB">Undo</span>
</p>
</div>
<svg xmlns='http://www.w3.org/2000/svg' id='soil' width='100%' height='24em'>
</svg>
<span class="logic" id="hintB">Hint</span><p id="hintT"></p>
<div id="winBar" style="visibility:hidden;">
<b>QED.</b>
<div style="text-align:center;">
<p>
<style>.winbutton{cursor:pointer;border:4px solid blue;border-radius:10px;padding:5px;margin:10px;font-size:400%}</style>
<span class="winbutton" id="againB">&#8635;</span>
<span class="winbutton" id="nextB">&#9654;</span>
</p>
</div>
</div>
<p id="postT"></p>
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

== See Also ==

 * http://teachinglogic.liglab.fr/DN/index.php[An automated theorem prover].

 * https://www.cs.cmu.edu/~fp/courses/atp/handouts/atp.pdf[Course notes on
 automated theorem proving].

 * https://www.winterdrache.de/freeware/domino/data/article.html[A domino-based natural deduction game].

 * https://en.wikipedia.org/wiki/Implicational_propositional_calculus[Implicational propositional calculus]. Along with a false proposition, the three rules &rArr;I , &rArr;E, and LEM suffice for classical propositional logic. LEM is equivalent to https://en.wikipedia.org/wiki/Peirce%27s_law[Peirce's law], which means we can remove false from the axioms completely.

 * http://math.andrej.com/2016/10/10/five-stages-of-accepting-constructive-mathematics/[Abandoning LEM makes sense for computer scientists].
 Roughly speaking, LEM means two wrongs make a right. For instance, if `f` is
 a function that takes an integer but is buggy then LEM is sort of claiming we
 can feed `f` to another buggy function to somehow cancel out the bugs and
 magically produce an integer!

== Ideas ==

 * Intuitionistic natural deduction satisfies the 'subformula property': a
 theorem can be proved solely from hypotheses that appear somewhere in the
 theorem.
+
We could add some levels where the theorem is shown as a tree, and the player
can select subtrees to create hypotheses.

 * Theorems proved in previous levels ought to be available in later levels.

 * We could add &forall; and &exist; for first-order logic. Or add
 &Pi; and &Sigma; types and move towards homotopy type theory. Or both!

 * Proof relevance: show the Haskell equivalent of a proof.

== A Haste Bug ==

We use `Data.Graph.Inductive.Graph` to store the proofs in progress.
Unfortunately,
https://github.com/valderman/haste-compiler/issues/113[a Haste bug] complicates
installing this package. Hopefully this will be fixed soon, but for now:

  1. Run `haste-cabal install fgl`. This fails.
  2. Extract the contents of the `fgl` package's archive.
  3. Remove all lines containing `ANN` from all source files.
  4. Re-archive the files and replace the original.
  5. Re-run `haste-cabal install fgl`.

\begin{code}
import Control.Concurrent hiding (readMVar)
import Control.Monad
import Haste.DOM
import Haste.Events
import Haste.Foreign
import Haste.JSString (pack)
import Data.Graph.Inductive.Graph hiding (Node)
import qualified Data.Graph.Inductive.Graph as G
import Data.Graph.Inductive.PatriciaTree
import Data.List
import Data.Maybe
import qualified Data.Map.Strict as M
import Data.Tree
import Control.Arrow
import Text.ParserCombinators.Parsec
\end{code}

== Tree drawing ==

We copy our link:../haskell/drawtree.html[tree drawing code], and extend it to handle
nodes of variable width, as well as nodes with only one child.

\begin{code}
data RT a = RT { xpos :: Int
               , shift :: Int
               , hold :: a
               , link :: Maybe (Int, RT a)
               , kids :: [RT a]
               } deriving Show

addx :: Int -> RT a -> Tree (Int, a)
addx i rt = Node (xpos rt + shift rt + i, hold rt) $
  addx (i + shift rt) <$> kids rt

padding :: Int  -- Minimum horizontal gap between nodes.
padding = 50

placeRT :: (a -> Int) -> Tree a -> RT a
placeRT _ (Node a [])         = RT 0 0 a Nothing []
placeRT width (Node a [k])    = RT (xpos r) 0 a Nothing [r]
  where r = placeRT width k
placeRT width (Node a [l, r]) = RT m 0 a Nothing xs where
  [ll, rr] = placeRT width <$> [l, r]
  g = padding - minimum (zipWith (-)
    (contour (((-1) *) . width) head (0, rr)) (contour width last (0, ll)))
  s = xpos ll + xpos rr
  gap = abs g + mod (abs g + s) 2  -- Adjust so midpoint is whole number.
  m = div (s + gap) 2
  xs = if g >= 0 then weave ll                 rr { shift = gap }
                 else weave ll { shift = gap } rr
placeRT _ _ = error "binary trees only please"

drawRT :: (a -> Int) -> Tree a -> Tree (Int, a)
drawRT width = addx 0 . placeRT width

contour :: (a -> Int) -> ([RT a] -> RT a) -> (Int, RT a) -> [Int]
contour width f (acc, rt) = h : case kids rt of
  [] -> maybe [] (contour width f . first (+ acc')) (link rt)
  ks -> contour width f (acc', f ks)
  where acc' = acc + shift rt
        h    = acc'+ xpos rt + width (hold rt)

weave :: RT a -> RT a -> [RT a]
weave l r = [weave' id (0, l) (0, r), weave' reverse (0, r) (0, l)]

weave' :: ([RT a] -> [RT a]) -> (Int, RT a) -> (Int, RT a) -> RT a
weave' f (accL, l) (accR, r)
  | Nothing      <- follow = l
  | Just (dx, x) <- link l = l { link = Just (dx, weave' f (dx + accL', x) y) }
  | (k:ks)   <- f $ kids l = l { kids = f $ weave' f (accL', k) y : ks }
  | otherwise              = l { link = first (+(-accL')) <$> follow }
  where
    accL' = accL + shift l
    accR' = accR + shift r
    follow | (k:_) <- f $ kids r = Just (accR', k)
           | otherwise           = first (accR' +) <$> link r
    Just y = follow
\end{code}

== Expression parser ==

In free play mode, the player can type in arbitrary hypotheses, which we
read using link:../haskell/parse.html[parser combinators]. The `OrHalf` is a
special value that the &or;-elimination rule puts in a placeholder node in
order to keep the tree binary.

\begin{code}
data Expr = Expr :-> Expr | Expr :& Expr | Expr :+ Expr
  | Var String | Bot | Top
  | OrHalf deriving Eq

instance Show Expr where
  show OrHalf    = "\x2228"
  show Bot       = "⊥"
  show Top       = "⊤"
  show (Var s)   = s
  show (x :-> y) = showL x ++ "\x21d2" ++ show y where
    showL e = case e of
      _ :-> _ -> "(" ++ show e ++ ")"
      _       -> show e
  show (x :& y) = showAnd x ++ "\x2227" ++ showAnd y where
    showAnd t@(_ :-> _) = "(" ++ show t ++ ")"
    showAnd t@(_ :+ _)  = "(" ++ show t ++ ")"
    showAnd t           = show t
  show (x :+ y) = showOr x ++ "\x2228" ++ showOr y where
    showOr t@(_ :-> _) = "(" ++ show t ++ ")"
    showOr t           = show t

proposition :: Parser Expr
proposition = do
  spaces
  r <- expr
  eof
  pure r
    where
    expr = atm `chainl1` and' `chainl1` or' `chainr1` imp
    atm = between (sp $ char '(') (sp $ char ')') expr <|> do
      s <- sp $ many1 alphaNum
      pure $ case s of
        "0" -> Bot
        "1" -> Top
        _   -> Var s
    imp = sp (string "->" <|> string "=>") >> pure (:->)
    and' = sp (string "&" <|> string "*") >> pure (:&)
    or' = sp (string "|" <|> string "+") >> pure (:+)
    sp :: Parser a -> Parser a  -- Eats trailing spaces.
    sp = (>>= (spaces >>) . pure)

parseProp :: String -> Expr
parseProp s = x where Right x = parse proposition "" s
\end{code}

== Levels ==

We look up a level's specifications from an association list.

A hidden element in the HTML file stores level titles, hints, and victory
messages.

[pass]
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
<div id="msgsDiv" display="none">
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

 * Show yourself!

   1. Click on the *a*, then on *⇒I*. This uses up a red node to make a new
   root node containing the (⇒) symbol.

   2. In English, the proof says: suppose "a" is true. Then "a" is true.
   Therefore "a" implies "a".
+
*⇒I* is the ⇒-'introduction' rule, which 'discharges' a 'hypothesis'
(a red node) by creating a new root node containing an implication from the
hypothesis to the old root node.
+
In this case, the old root node is also the hypothesis.
+
This is the type of the I combinator; Haskell's `id` function.

 * Ghosts of departed hypotheses

   1. Select the two nodes in the right order and apply *⇒I* to use up
a red node. It disappears because it is alone. Then apply *⇒I* again.

   2. English: suppose "a" is true. Then "b" implies "a". Therefore
"a" implies "b" implies "a".
+
The disappearing red node represents discharging zero copies of a hypothesis.
+
The *⇒* operator is right-associative, that is, *a⇒b⇒c* means *a⇒(b⇒c)*.
+
This is the type of the K combinator; Haskell's `const` function.

 * An introduction to elimination

   1. Use *⇒E*, then apply *⇒I* to the hypotheses in the right order.

   2. *⇒E* is the ⇒-'elimination' rule, affectionately known as 'modus ponens'.
+
This is the type of the reverse apply operator; the `(&)` function in Haskell's
`Data.Function`.

 * Take me to your Reader

   1. Three doses of *⇒E* followed by three doses of *⇒I*.

   2. Multiple copies of a hypothesis can be discharged at once.
+
This is the type of the S combinator; the function `ap` in Haskell's Reader
monad.

 * Two wrongs make a right

   1. After *⇒E*, apply *LEM* to the correct hypothesis. The rest is forced.

   2. *⊥* represents falsehood.
+
*LEM* is the 'law of the excluded middle', or 'tertium non datur', and
equivalent to 'proof by contradiction', or 'reductio ad absurdum'. It is
the hallmark of 'classical logic'.
+
Some view *¬* as a primitive logical operator. For us, *¬a* is a synonym for
*a⇒⊥*.

 * The puff of logic

   1. After a certain step, follow the same steps as the previous level.

   2. The 'principle of explosion' or 'ex falso quodlibet'. The `absurd` function of Haskell's `Data.Void`.

 * Aftershock

   1. After a certain step, follow the same steps as the previous level.

   2. We'll need this later.

 * The lie becomes the truth

   1. Start with a couple of eliminations.

   2. https://en.wikipedia.org/wiki/Law_of_excluded_middle['Principia Mathematica' calls this the complement of 'reductio ad absurdum' (pp. 103-104)].

 * Holding the Peirce strings

   1. Redo the steps for the previous level, and combine with steps from
the levels before.

   2. Peirce's law.

 * One rule to rule them all

   1. Good luck!

   2. https://www.jstor.org/stable/20488489[Łukasiewicz found this single axiom schema is all we need for a Hilbert-style classical propositional calculus].

 * Aid and abet

   1. Choose the appropriate ∧-'elimination' rule; press 1 to keep
   the left side and 2 to keep the right.

   2. Redundant conjunctions in law are called https://en.wikipedia.org/wiki/Legal_doublet['legal doublets'].

 * Caboodle and kit

   1. Use *∧1E* on one and *∧2E* on the other, then *∧I*.

   2. Why do we prefer "kit and caboodle?"
   https://www.youtube.com/watch?v=9GubdYZPYPg[Steven Pinker explains (around minute 34)].

 * Or else what?

   1. The ∨-'introduction' rules work similarly to *⇒I* from a lone
   hypothesis. Be mindful of the order the nodes are selected.

   2. Up until now, the rules of natural deduction really do seem natural.
   But how might ∨-'elimination' fit in?

 * Shine or rain

   1. Form two trees with root node *b∨a*, then apply *∨E*. This rule
   requires 3 input nodes.

   2. We only draw binary trees so the result of *∨E* looks a bit odd.

 * You say either and I say disjunction

   1. Apply *⇒E* twice, then *∨E*.

   2. This is the type of `either` in Haskell. After working through proofs
   like this, we understand why
   http://www.paultaylor.eu/stable/Proofs+Types.html[Girard says ∨-elimination
   is ``very bad''].

 * Logic bomb

   1. Use *⊥E*.

   2. Instead of LEM, intuitionistic logic provides the weaker *⊥E* rule.
   This is the 'principle of explosion', or 'ex falso quodlibet'.
+
Without *⊥E*, we have 'minimal logic'.

 * Two wrongs no longer make a right

   1. Outline: *a* leads to *a⇒⊥* which leads to the theorem.

   2. Without *LEM*, we can show *¬¬(a∨¬a)* but not *a∨¬a*.

 * Disjunction to implication

   1. Show *a* leads to the right-hand side, and similarly for *b*.
   Then use *∨E*.

   2. In classical logic, we can define *∨* in terms of *⇒* thanks
   to this theorem and its converse.

 * One-way trip

   1. Show *a* leads to a contradiction in the presence of
   *(((a⇒b)⇒b)⇒a∨b)⇒⊥*. Use *⊥E* with *b*. Then apply *⇒I* to discharge *a* and
   continue on to *((a⇒b)⇒b)⇒a∨b*. The rest is straightforward.

   2. The theorems *a∨b* and *(a⇒b)⇒b* are equivalent in the
   presence of LEM, but without it, we can only go in one direction.

 * Not to be confused with Distributism

   1. Show *a∧b* leads to *a∧(b∨c)*. Do the same for *a∧c*.

   2. One half of the many https://en.wikipedia.org/wiki/Distributive_property#Propositional_logic[distributive laws of propositional calculus].

 * The proof is always easier on the other side.

   1. Show *a∧(b∨c)* and *b* lead to the right-hand side. Repeat
   for *a∧(b∨c)* and *c*.

   2. The converse of the previous theorem.

[pass]
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
</div>
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

\begin{code}
data Level = Level Expr [Expr] String | FreePlay

getLevel :: Int -> Level
getLevel n
  | Just ((goal, hs), rules) <- lookup n lvls
    = Level (parseProp goal) (parseProp <$> hs) rules
  | otherwise = FreePlay
  where
  lvls = zip [1..] $
    [ (("a->a", ["a"]), "impliesI")
    , (("a->b->a", ["a", "b"]), "impliesI")
    , (("a->(a->b)->b", ["a", "a->b"]), "impliesI impliesE")
    , (("(a->b->c)->(a->b)->a->c", ["a","a","a->b","a->b->c"]), "impliesI impliesE")
    ] ++ (`zip` (repeat "classy"))
    [ ("((a->0)->0)->a", ["a->0","(a->0)->0"])
    , ("0->a", ["0","a->0","a->0"])
    , ("(a->0)->a->b", ["a","a->0","b->0","b->0"])
    , ("((a->0)->a)->a", ["a->0","a->0","(a->0)->a"])
    , ("((a->b)->a)->a", ["a","a->0","a->0","b->0","b->0","(a->b)->a"])
    , ("((a->b)->c)->((c->a)->(d->a))", ["a","a->0","a->0","b->0","b->0","d","c->a","(a->b)->c"])
    ] ++
    [ (("a&b->b", ["a&b"]), "impliesI impliesE and1E and2E")
    , (("a&b->b&a", ["a&b","a&b"]), "impliesI impliesE andI and1E and2E")
    , (("a->a+b", ["a","b"]), "impliesI impliesE or1I or2I orE")
    , (("a+b->b+a", ["a+b","a","a","b","b"]), "impliesI impliesE or1I or2I orE")
    , (("(a->c)->(b->c)->a+b->c", ["a", "b", "a->c", "b->c", "a+b"]), "impliesI impliesE or1I or2I orE")
    ] ++ (`zip` (repeat "intu"))
    [ ("0->a", ["0", "a"])
    , ("(a+(a->0)->0)->0", ["a","a","a->0","a+(a->0)->0","a+(a->0)->0"])
    , ("a+b->(a->b)->b", ["a+b","a","b","a->b","a->b"])
    , ("((((a->b)->b)->a+b)->0)->0", ["a","a","b","b","(a->b)->b","(a->b)->b","(((a->b)->b)->a+b)->0","(((a->b)->b)->a+b)->0"])
    , ("a&b+a&c->a&(b+c)", ["b","c","a&b","a&b","a&c","a&c","a&b+a&c"])
    , ("a&(b+c)->a&b+a&c", ["b","c","a&b","a&c","a&(b+c)","a&(b+c)","a&(b+c)"])
    ]
\end{code}

== More Haste workarounds ==

The `Haste.DOM` functions handle SVG poorly, so we provide workarounds.

The `readMVar` function causes compile errors so we supply our own.

\begin{code}
newElemSVG :: String -> IO Elem
newElemSVG = ffi $ pack $
  "(x => document.createElementNS('http://www.w3.org/2000/svg', x))"

getChildrenSVG :: Elem -> IO [Elem]
getChildrenSVG = ffi $ pack $ concat ["(function(e){var ch = [];",
  "for(e = e.firstChild; e != null; e = e.nextSibling)",
  "{ch.push(e);}return ch;})"]

readMVar :: MVar a -> IO a
readMVar v = do
  x <- takeMVar v
  putMVar v x
  pure x
\end{code}

== Tedium ==

Clunky UI code unfortunately intertwined with the rules of inference.

A proof in progress is a forest along with a record of all discharges.
We represent the forest with an inductive graph which is well-suited for
handling player clicks on any node.

With each edge, we associate an SVG line as well as an integer so out-edges
are ordered.

\begin{code}
type Grx = Gr (Elem, Expr) (Elem, Int)
type Dis = M.Map G.Node G.Node
type Proof = (Grx, Dis)

halfWidth :: Expr -> Int
halfWidth x = 6 * (length $ show x)

rootOf :: Grx -> G.Node -> G.Node
rootOf g h = case pre g h of
  [] -> h
  [t] -> rootOf g t
  _ -> undefined

foliage :: Grx -> G.Node -> [G.Node]
foliage g y | null ss   = [y]
            | otherwise = concatMap (foliage g) ss
            where ss = suc g y

nodeElem :: Grx -> G.Node -> Elem
nodeElem g = fst . fromJust . lab g

nodeExpr :: Grx -> G.Node -> Expr
nodeExpr g = snd . fromJust . lab g

xbounds :: Grx -> Tree (Int, G.Node) -> (Int, Int)
xbounds g (Node (x, n) ks) = foldl1' mm $ (x - w, x + w) : (xbounds g <$> ks)
  where
  w = halfWidth $ nodeExpr g n
  mm = uncurry (***) . (min *** max)

drawGr :: Grx -> G.Node -> Tree (Int, G.Node)
drawGr g node = drawRT (halfWidth . nodeExpr g) $ fromGraph node where
  fromGraph n = Node n $ fromGraph . fst <$> sortOn (snd . snd) (lsuc g n)

discharge :: Grx -> G.Node -> G.Node -> Dis -> Dis
discharge g x y dis = M.union dis $ M.fromList $ zip xs $ repeat y where
  xs = filter ((== nodeExpr g x) . nodeExpr g) (foliage g $ rootOf g x)

elemjs :: Elem -> String -> IO ()
elemjs e s = (ffi $ pack s :: Elem -> IO ()) e

defaultNode :: Elem -> IO ()
defaultNode e = do
  setStyle e "cursor" ""
  elemjs e "e => e.childNodes[0].setAttribute('stroke','grey')"
  elemjs e "e => e.childNodes[0].setAttribute('stroke-dasharray','1,1')"
  elemjs e "e => e.childNodes[1].removeAttribute('fill')"

hypNode :: Elem -> IO ()
hypNode e = do
  setStyle e "cursor" "pointer"
  elemjs e "e => e.childNodes[1].setAttribute('fill','red')"
  elemjs e "e => e.childNodes[0].removeAttribute('stroke-dasharray')"
  elemjs e "e => e.childNodes[0].setAttribute('stroke','blue')"

rootNode :: Elem -> IO ()
rootNode e = do
  setStyle e "cursor" "pointer"
  elemjs e "e => e.childNodes[0].setAttribute('stroke','black')"
  elemjs e "e => e.childNodes[0].removeAttribute('stroke-dasharray')"
  elemjs e "e => e.childNodes[0].setAttribute('stroke','blue')"

draw :: Elem -> Int -> Grx -> Int -> Tree (Int, G.Node) -> IO ()
draw soil x0 g y (Node (x, n) ks) = do
  setAttr (nodeElem g n) "transform" $ "translate" ++ show (x0 + x, -40*y)
  mapM_ (draw soil x0 g $ y + 1) ks

enableRule :: Elem -> IO ()
enableRule e = do
  setStyle e "border" "2px solid blue"
  setStyle e "color" "black"

disableRule :: Elem -> IO ()
disableRule e = do
  setStyle e "border" "1px dotted grey"
  setStyle e "color" "grey"

selectNode :: Elem -> IO ()
selectNode = ffi $ pack $ "e => e.childNodes[0].setAttribute('fill','yellow')"

ageNode :: Elem -> IO ()
ageNode = ffi $ pack $ "e => e.childNodes[0].setAttribute('fill','orange')"

deselectNode :: Elem -> IO ()
deselectNode = ffi $ pack $ "e => e.childNodes[0].setAttribute('fill','white')"

theorem :: Proof -> Maybe Expr
theorem (g, dis) | null live, [t] <- tips = Just $ nodeExpr g t
                 | otherwise              = Nothing
                 where
  live = filter (`M.notMember` dis) $ filter (null . suc g) $ nodes g
  tips = filter (null . pre g) $ nodes g

mustEldest :: Elem -> IO Elem
mustEldest = fmap fromJust . getFirstChild

main :: IO ()
main = withElems
  [ "soil", "winBar", "ruleBar", "hypoT", "newHypoB", "errT", "msgsDiv"
  , "impliesI", "impliesE", "notNot"
  , "andI", "and1E", "and2E", "or1I", "or2I", "orE", "falseE"
  , "againB", "nextB", "hintB", "hintT", "undoB", "hypoDiv", "preT", "postT"] $
    \[ soil, winBar, ruleBar, hypoT, newHypoB, errT, msgsDiv
     , impliesI, impliesE, notNot
     , andI, and1E, and2E, or1I, or2I, orE, falseE
     , againB, nextB, hintB, hintT, undoB, hypoDiv, preT, postT] -> do
  msgsDivKids <- getChildren =<< mustEldest =<< mustEldest msgsDiv
  msgs <- forM msgsDivKids $ \e -> do  -- Title, hint, victory message.
    ol <- getChildren =<< mustEldest . (!!1) =<< getChildren e
    eldest <- mustEldest e
    mapM (`getProp` "innerHTML") $ eldest : ol
  let
    classicRules = [impliesI, impliesE, notNot]
    intuRules =
      [impliesI, impliesE, andI, and1E, and2E, or1I, or2I, orE, falseE]
    allRules = classicRules ++ intuRules
  sel <- newMVar []
  proof <- newMVar (mkGraph [] [], M.empty)
  acts <- newMVar []
  history <- newMVar []
  level <- newMVar 1
  let
    resetActs = swapMVar acts [] >>=
      mapM_ (\(e, h) -> disableRule e >> unregisterHandler h)
    nodeClick i = do
      (g, dis) <- readMVar proof
      when (null (suc g i) && M.notMember i dis || null (pre g i)) $ select g i

    activate :: Elem -> (Proof -> IO Proof) -> IO ()
    activate e f = do
      enableRule e
      h <- e `onEvent` Click $ const $ do
        resetActs
        prf <- f =<< load
        save prf
        case theorem prf of
          Just t -> do
            n <- readMVar level
            case getLevel n of
              Level goal _ _ -> if t == goal then do
                  setProp postT "innerHTML" $ msgs!!(n - 1)!!2
                  setStyle winBar "visibility" "visible"
                  setStyle ruleBar "display" "none"
                  setStyle hintB "display" "none"
                else
                  setProp postT "innerHTML" $ "Proved " ++ show t
                    ++ " but we want " ++ show goal
              _ -> setProp postT "innerHTML" $ "Proved " ++ show t
          _ -> setProp postT "innerHTML" ""
      modifyMVar_ acts $ pure . ((e, h):)

    ghost x y t (g, dis) = do
      deleteChild soil $ nodeElem g x
      g1 <- delNode x . snd <$> newProp g [y] t
      pure (g1, dis)

    dSpawn x y t (g, dis) = do
      (k, g1) <- newProp g [y] t
      pure (g1, discharge g1 x k dis)

    spawn ks t (g, dis) = do
      g1 <- snd <$> newProp g ks t
      pure (g1, dis)

    select g i = do
      let e = nodeElem g i
      xs0 <- takeMVar sel
      xs <- if elem i xs0
        then do
          deselectNode e
          case delete i xs0 of
            [] -> pure []
            xs1@(h:_) -> do
              selectNode $ nodeElem g h
              pure xs1
        else do
          forM_ xs0 $ ageNode . nodeElem g
          selectNode e
          pure $ i:xs0
      putMVar sel xs
      resetActs
      let
        exprOf = nodeExpr g
        isRoot = null . pre g
        isLeaf = null . suc g
        impi
          | [y, x] <- xs, isLeaf x, isRoot x, isRoot y
            = [(impliesI, ghost x y $ exprOf x :-> exprOf y)]
          | [x] <- xs, isLeaf x, y <- rootOf g x
            = [(impliesI, dSpawn x y $ exprOf x :-> exprOf y)]
          | [y] <- filter isRoot xs, [x] <- delete y xs, rootOf g x == y
            = [(impliesI, dSpawn x y $ exprOf x :-> exprOf y)]
          | otherwise = []
        mopo
          | [x, y] <- xs, isRoot x, isRoot y, a :-> b <- exprOf y, a == exprOf x
            = [(impliesE, spawn [x, y] b)]
          | [x, y] <- xs, isRoot x, isRoot y, a :-> b <- exprOf x, a == exprOf y
            = [(impliesE, spawn [y, x] b)]
          | otherwise = []
        lem
          | ([y], [x]) <- partition isRoot xs, y == rootOf g x,
            Bot <- exprOf y, t :-> Bot <- exprOf x
            = [(notNot, dSpawn x y t)]
          | [x] <- xs, isLeaf x, y <- rootOf g x,
            Bot <- exprOf y, t :-> Bot <- exprOf x
            = [(notNot, dSpawn x y t)]
          | otherwise = []

        conj
          | [y, x] <- xs, isRoot x, isRoot y
            = [(andI, spawn [x, y] $ exprOf x :& exprOf y)]
          | [x] <- xs, isRoot x, l :& r <- exprOf x
            = [(and1E, spawn [x] l), (and2E, spawn [x] r)]
          | otherwise = []
        disj
          | [y, x] <- xs, isLeaf x, isRoot x, isRoot y
            = [(or1I, ghost x y $ exprOf y :+ exprOf x),
               (or2I, ghost x y $ exprOf x :+ exprOf y)]
          | length xs == 3 = take 1 $ concatMap orCheck $ permutations xs
          | otherwise      = []
        orCheck [x, y, z]
          | isLeaf x, isLeaf y, rx <- rootOf g x, ry <- rootOf g y,
            exprOf x :+ exprOf y == exprOf z, t <- exprOf rx,
            rx /= ry, t == exprOf ry
            = [(orE, \(g0, dis0) -> do
          (k, g1) <- newProp g0 [rx, ry] OrHalf
          g2 <- snd <$> newProp g1 [k, z] t
          pure (g2, discharge g0 y k $ discharge g0 x k dis0))]
        orCheck _ = []
        efq
          | [y, x] <- xs, Bot <- exprOf y = [(falseE, ghost x y $ exprOf x)]
          | [x, y] <- xs, Bot <- exprOf y = [(falseE, ghost x y $ exprOf x)]
          | otherwise = []
      mapM_ (uncurry activate) $ concat [impi, mopo, lem, conj, disj, efq]

    -- | Reads proof from MVar. Records it to undo history. Clears selection.
    load = do
      enableRule undoB
      p@(g, _) <- takeMVar proof
      swapMVar sel [] >>= mapM_ (deselectNode . nodeElem g)
      modifyMVar_ history $ pure . (p:)
      pure p

    translate :: Parser (String, String)
    translate = do
      void $ string "translate("
      x <- many1 (digit <|> char '-')
      void $ string ","
      y <- many1 (digit <|> char '-')
      void $ string ")"
      pure (x, y)

    -- | Saves proof to MVar. Draws it.
    save (g, dis) = do
      putMVar proof (g, dis)
      let
        f [] d acc = pure (acc, d)
        f (r:rs) d acc = do
          draw soil (acc - x0) g 0 drawing
          f rs (max d dep) $ acc + x1 - x0 + 25
          where
            drawing = drawGr g r
            (x0, x1) = xbounds g drawing
            dep = length $ levels drawing
      (bx1, by1) <- f (filter (null . pre g) $ nodes g) 0 0
      forM_ (labEdges g) $ \(i, j, (l, _)) -> do
        Right (x1, y1) <- parse translate "" <$> getAttr (nodeElem g i) "transform"
        Right (x2, y2) <- parse translate "" <$> getAttr (nodeElem g j) "transform"
        setAttr l "x1" x1
        setAttr l "y1" y1
        setAttr l "x2" x2
        setAttr l "y2" y2
      forM_ (nodes g) $ \i -> let e = nodeElem g i in
        when (nodeExpr g i /= OrHalf) $ do
          defaultNode e
          when (null (suc g i) && M.notMember i dis) $ hypNode e
          when (null $ pre g i) $ rootNode e
      setAttr soil "viewBox" $ "-5 " ++ show (-40 * by1) ++ " "
        ++ show (bx1) ++ " " ++ show (40*by1 + 40)

    newProp g qs t = do
      let k = if isEmpty g then 0 else snd (nodeRange g) + 1
      e <- newElemSVG "g"
      let w = halfWidth t
      er <- newElemSVG "rect" `with`
        [ attr "width"  =: show (2 * w)
        , attr "height" =: "24"
        , attr "fill"   =: "white"
        , attr "stroke" =: "black"
        , attr "x"      =: show (-w)
        , attr "y"      =: "-12"
        , attr "rx"     =: "4"
        , attr "ry"     =: "4"
        ]
      when (t == OrHalf) $ do
        setAttr er "stroke" "none"
        setAttr er "fill" "none"
      appendChild e er
      when (t /= OrHalf) $ do
        et <- newElemSVG "text" `with`
          [ attr "x" =: "0"
          , attr "y" =: "0"
          , attr "text-anchor" =: "middle"
          , attr "alignment-baseline" =: "central"
          , prop "textContent" =: show t
          ]
        appendChild e et
      appendChild soil e
      void $ e `onEvent` Click $ const $ nodeClick k
      ls <- replicateM (length qs) $ newElemSVG "line" `with`
        [ attr "stroke" =: "black" ]
      ks <- getChildrenSVG soil
      setChildren soil $ ls ++ ks
      pure (k, insEdges [(k, q, ln) | (q, ln) <- zip qs $ zip ls [0..]] $
        insNode (k, (e, t)) g)

    addHypo t = do
      (g0, dis0) <- load
      (k, g1) <- newProp g0 [] t
      let dis1 = if t == Top then M.insert k k dis0 else dis0
      save (g1, dis1)

  void $ undoB `onEvent` Click $ const $ do
    hist <- readMVar history
    case hist of
      [] -> pure ()
      ((g, dis):rest) -> do
        void $ load
        let
          ls = (\(_, _, (l, _)) -> l) <$> labEdges g
          ks = fst . snd <$> labNodes g
        setChildren soil $ ls ++ ks
        setProp errT "innerHTML" ""
        setProp postT "innerHTML" ""
        save (g, dis)
        when (null rest) $ disableRule undoB
        void $ swapMVar history rest

  void $ newHypoB `onEvent` Click $ const $ do
    s <- getProp hypoT "value"
    case parse proposition "" s of
      Right x -> do
        setProp errT "innerHTML" ""
        addHypo x
      Left err -> setProp errT "innerHTML" $ show err

  let
    ruleset "intu"   = pure intuRules
    ruleset "classy" = pure classicRules
    ruleset names    = catMaybes <$> mapM elemById (words names)
    setup n = do
      forM_ allRules disableRule
      void $ swapMVar proof (mkGraph [] [], M.empty)
      void $ swapMVar sel []
      clearChildren soil
      setStyle winBar "visibility" "hidden"
      setStyle ruleBar "display" ""
      case getLevel n of
        Level goal hs rules -> do
          setStyle hintB "display" ""
          setProp hintT "innerHTML" ""
          forM_ allRules $ \e -> setStyle e "display" "none"
          ruleset rules >>= mapM_ (\e -> setStyle e "display" "")
          mapM_ addHypo hs
          setProp preT "innerHTML" $ concat
            [ "<p>Level "
            , show n
            , ":<i>"
            , msgs!!(n - 1)!!0
            , "</i></p><p><b>Theorem</b>:"
            , "<div style='text-align:center;font-size:150%'><span>"
            , show goal
            , "</span></div>"
            , "</p><b>Proof:</b>"
            ]
          setProp postT "innerHTML" ""
        FreePlay -> do
          setStyle hintB "display" "none"
          setProp hintT "innerHTML" ""
          forM_ allRules $ \e -> setStyle e "display" ""
          setStyle hypoDiv "display" ""
          setProp preT "innerHTML" "<h2>Free Play</h2><p>Type '=>' or '->' for implication, '0' and '1' for false and true.</p>"
          setProp postT "innerHTML" ""
      disableRule undoB
      void $ swapMVar history []
  void $ nextB `onEvent` Click $ const $ do
    n <- takeMVar level
    putMVar level $ n + 1
    setup $ n + 1
  void $ againB `onEvent` Click $ const $ setup =<< readMVar level
  void $ hintB `onEvent` Click $ const $ do
    n <- readMVar level
    setStyle hintB "display" "none"
    setProp hintT "innerHTML" $ msgs!!(n - 1)!!1
  setup =<< readMVar level
\end{code}
