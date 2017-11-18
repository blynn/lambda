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
<style>.logic{cursor:pointer;border:1px solid black;padding:5px;margin:5px;}</style>
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
<button style="float:right;" id="undoB">Undo</button>
</p>
</div>
<svg xmlns='http://www.w3.org/2000/svg' id='soil' width='100%' height='32em'>
</svg>
<p id="postT"></p>
<div style="text-align:center;">
<button id="nextB" style="visibility:hidden;font-size:400%;">&#9654;</button>
</div>
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

*Level 1*: The moves are forced. We must use the &rArr;-'introduction' rule, or
&rArr;I for short, which 'discharges' a 'hypothesis'.

This rule uses up a red node (hypothesis) to make a new root node containing
the (&rArr;) symbol. On the left, it places the contents of the red node,
and on the right, it places the contents of the old root node.

Here, the root node is also the hypothesis.

*Level 2*: Select the two nodes in the right order before
applying &rArr;I. Then apply &rArr;I again.

Here, we discharge zero copies of a hypothesis. That is, we use up a lone red
node, which our game then removes to reduce clutter.

*Level 3*: Use the &rArr;-'elimination' rule first, or &rArr;E for short, which
is affectionately called 'modus ponens'. Then apply &rArr;I to the hypotheses
in the right order.

*Level 4*: Three doses of &rArr;E followed by three doses of &rArr;I.
Observe multiple copies of a hypothesis can be discharged at once.

*Level 5*: After &rArr;E, apply LEM, the 'law of the excluded middle',
or 'terium non datur' to the correct hypothesis. The rest is forced.
Other presentations of logic may refer to this rule as 'proof by contradiction',
or 'reductio ad absurdum'.

*Level 6*: After a certain step, follow the same steps as the previous level.

*Level 7*: The same as the previous level, with a few more steps.

*Level 8*: Also the same as level 6, with a few more steps.

*Level 9*: Redo the steps for the previous level, and combine with steps from
the levels before.

*Level 10*: Good luck!

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
read using link:../haskell/parse.html[parser combinators]. The `OrHalf` is a special value
that the &or;-elimination rule puts in a placeholder node in order to keep the
tree binary.

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

\begin{code}
data Level = Level Expr [Expr] String Bool | FreePlay

getLevel :: Int -> Level
getLevel n
  | Just (intu, (goal, hs, comment)) <- lookup n lvls
    = Level (parseProp goal) (parseProp <$> hs) comment intu
  | otherwise = FreePlay
  where
  lvls = zip [1..] $ clvls ++ ilvls
  clvls = zip (repeat False)
    [ ("a->a", ["a"], "The I combinator. The `id` function of Haskell.")
    , ("a->b->a", ["a", "b"], "The K combinator. The `const` function of Haskell.")
    , ("a->(a->b)->b", ["a", "a->b"], "The reverse apply operator. The `(&amp;)` function in Haskell's `Data.Function`.")
    , ("(a->b->c)->(a->b)->a->c", ["a","a","a->b","a->b->c"], "The S combinator. The function `ap` in Haskell's Reader monad.")
    , ("((a->0)->0)->a", ["a->0","(a->0)->0"], "Classical logic's <i>reductio ad absurdum</i> or <i>proof by contradiction</i>.")
    , ("0->a", ["0","a->0","a->0"], "The <i>principle of explosion</i> or <i>ex falso quodlibet</i>. The `absurd` function of Haskell's `Data.Void`.")
    , ("(a->0)->a->b", ["a","a->0","b->0","b->0"], "")
    , ("((a->0)->a)->a", ["a->0","a->0","(a->0)->a"], "")
    , ("((a->b)->a)->a", ["a","a->0","a->0","b->0","b->0","(a->b)->a"], "Peirce's law.")
    , ("((a->b)->c)->((c->a)->(d->a))", ["a","a->0","a->0","b->0","b->0","d","c->a","(a->b)->c"], "Łukasiewicz found this single axiom schema is all we need for a Hilbert-style classical propositional calculus.")
    ]
  ilvls = zip (repeat True)
    [ ("0->a", ["0", "a"], "Ex falso quodlibet.")
    , ("a&b->b&a", ["a&b", "a&b"], "")
    , ("a+b->b+a", ["a+b", "a", "a", "b", "b"], "")
    , ("a&b+a&c->a&(b+c)", ["b","c","a&b","a&b","a&c","a&c","a&b+a&c"], "")
    , ("a&(b+c)->a&b+a&c", ["a","a","b","c","a&b","a&c","b+c","a&(b+c)","a&(b+c)"], "")
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
  xs = filter ((== nodeExpr g x) . nodeExpr g) (foliage g y)

chargeElem :: Elem -> IO ()
chargeElem = ffi $ pack $ "e => e.childNodes[1].setAttribute('fill','red')"

dischargeElem :: Elem -> IO ()
dischargeElem = ffi $ pack $ "e => e.childNodes[1].removeAttribute('fill')"

draw :: Elem -> Int -> Grx -> Int -> Tree (Int, G.Node) -> IO ()
draw soil x0 g y (Node (x, n) ks) = do
  setAttr (nodeElem g n) "transform" $ "translate" ++ show (x0 + x, -40*y)
  mapM_ (draw soil x0 g $ y + 1) ks

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

main :: IO ()
main = withElems
  [ "soil", "ruleBar", "hypoT", "newHypoB", "errT"
  , "impliesI", "impliesE", "notNot"
  , "andI", "and1E", "and2E", "or1I", "or2I", "orE", "falseE"
  , "nextB", "undoB", "hypoDiv", "preT", "postT"] $
    \[ soil, ruleBar, hypoT, newHypoB, errT
     , impliesI, impliesE, notNot
     , andI, and1E, and2E, or1I, or2I, orE, falseE
     , nextB, undoB, hypoDiv, preT, postT] -> do
  let
    classicRules = [impliesI, impliesE, notNot]
    intuRules =
      [impliesI, impliesE, andI, and1E, and2E, or1I, or2I, orE, falseE]
    allRules = classicRules ++ intuRules
  forM_ allRules disableRule
  forM_ allRules $ \e -> setStyle e "display" "none"
  forM_ classicRules $ \e -> setStyle e "display" "initial"
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
      setStyle e "border" "2px solid black"
      setStyle e "color" "black"
      h <- e `onEvent` Click $ const $ do
        resetActs
        prf <- f =<< load
        save prf
        case theorem prf of
          Just t -> do
            lvl <- getLevel <$> readMVar level
            case lvl of
              Level goal _ comment _ -> if t == goal then do
                  setProp postT "innerHTML" $ "<p><b>QED.</b></p>" ++ comment
                  setStyle nextB "visibility" "visible"
                  setStyle ruleBar "display" "none"
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
          | isLeaf x, isLeaf y,
            exprOf x :+ exprOf y == exprOf z, t <- exprOf (rootOf g x),
            t == exprOf (rootOf g y)
            = [(orE, \(g0, dis0) -> do
          (k, g1) <- newProp g0 [rootOf g x, rootOf g y] OrHalf
          g2 <- snd <$> newProp g1 [k, z] t
          pure (g2, discharge g1 y k $ discharge g1 x k dis0))]
        orCheck _ = []
        efq
          | [y, x] <- xs, Bot <- exprOf y = [(falseE, ghost x y $ exprOf x)]
          | [x, y] <- xs, Bot <- exprOf y = [(falseE, ghost x y $ exprOf x)]
          | otherwise = []
      mapM_ (uncurry activate) $ concat [impi, mopo, lem, conj, disj, efq]

    -- | Reads proof from MVar. Records it to undo history. Clears selection.
    load = do
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
      forM_ (filter (null . suc g) $ nodes g) $ \i ->
        (if M.member i dis then dischargeElem else chargeElem) $ nodeElem g i
      setAttr soil "viewBox" $ "-5 " ++ show (-40 * by1) ++ " "
        ++ show (bx1) ++ " " ++ show (40*by1 + 40)

    newProp g qs t = do
      let k = if isEmpty g then 0 else snd (nodeRange g) + 1
      e <- newElemSVG "g"
      let w = halfWidth t
      appendChild e =<< newElemSVG "rect" `with`
        [ attr "width"  =: show (2 * w)
        , attr "height" =: "24"
        , attr "fill"   =: "white"
        , attr "stroke" =: "black"
        , attr "x"      =: show (-w)
        , attr "y"      =: "-12"
        ]
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
        save (g, dis)
        void $ swapMVar history rest

  void $ newHypoB `onEvent` Click $ const $ do
    s <- getProp hypoT "value"
    case parse proposition "" s of
      Right x -> do
        setProp errT "innerHTML" ""
        addHypo x
      Left err -> setProp errT "innerHTML" $ show err

  let
    setup n = do
      void $ swapMVar proof (mkGraph [] [], M.empty)
      clearChildren soil
      setStyle nextB "visibility" "hidden"
      setStyle ruleBar "display" "initial"
      case getLevel n of
        Level goal hs _ intu -> do
          forM_ allRules $ \e -> setStyle e "display" "none"
          forM_ (if intu then intuRules else classicRules) $
            \e -> setStyle e "display" "initial"
          mapM_ addHypo hs
          setProp preT "innerHTML" $ "<h2>Level " ++ show n ++ "</h2><p><b>Theorem</b>: " ++ show goal ++ "</p><b>Proof:</b>"
          setProp postT "innerHTML" ""
        FreePlay -> do
          forM_ allRules $ \e -> setStyle e "display" "initial"
          setStyle hypoDiv "display" "initial"
          setProp preT "innerHTML" "<h2>Free Play</h2><p>Type '=>' or '->' for implication, '0' and '1' for false and true.</p>"
          setProp postT "innerHTML" ""
      void $ swapMVar history []

  void $ nextB `onEvent` Click $ const $ do
    n <- takeMVar level
    putMVar level $ n + 1
    setup $ n + 1
  setup =<< readMVar level
\end{code}
