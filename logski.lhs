= λ to SKI, logarithmically =

[pass]
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
<script src="logski.js"></script>
<textarea id="input" rows="5" cols="80">
\f -> (\x -> x x)(\x -> f(x x))
</textarea>
<br>
<button id="skiB">Convert</button>
<br>
<textarea id="lin" rows="4" cols="80" readonly></textarea>
<br>
<textarea id="log" rows="8" cols="80" readonly></textarea>
<br>
</script>
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Oleg Kiselyov describes
http://okmij.org/ftp/tagless-final/ski.pdf[a linear-time and linear-space
algorithm for converting lambda terms to combinators]:

\begin{code}
{-# LANGUAGE CPP #-}
#ifdef __HASTE__
{-# LANGUAGE OverloadedStrings #-}
import Haste.DOM
import Haste.Events
#endif
import Data.List
import Text.Parsec

data Deb = Zero | Succ Deb | Lam Deb | App Deb Deb deriving Show
infixl 5 :#
data Com = Com :# Com | S | I | C | K | B | Sn Int | Bn Int | Cn Int
ski :: Deb -> (Int, Com)
ski deb = case deb of
  Zero                           -> (1,       I)
  Succ d    | x@(n, _) <- ski d  -> (n + 1,   f (0, K) x)
  App d1 d2 | x@(a, _) <- ski d1
            , y@(b, _) <- ski d2 -> (max a b, f x y)
  Lam d | (n, e) <- ski d -> case n of
                               0 -> (0,       K :# e)
                               _ -> (n - 1,   e)
  where
  f (a, x) (b, y) = case (a, b) of
    (0, 0)             ->         x :# y
    (0, n)             -> Bn n :# x :# y
    (n, 0)             -> Cn n :# x :# y
    (n, m) | n == m    -> Sn n :# x :# y
           | n < m     ->                Bn (m - n) :# (Sn n :# x) :# y
           | otherwise -> Cn (n - m) :# (Bn (n - m) :#  Sn m :# x) :# y
\end{code}

It relies on memoizing the bulk variants of the B, C, and S combinators:

------------------------------------------------------------------------------
Bn n f g xn ... x1 =  f            (g xn ... x1)
Cn n f g xn ... x1 = (f xn ... x1)  g
Sn n f g xn ... x1 = (f xn ... x1) (g xn ... x1)
------------------------------------------------------------------------------

In particular, `B 1`, `C 1`, and `S 1` are the standard `B`, `C`, and `S`
combinators.

Linear complexity is implied by:

\begin{code}
linBulk :: Com -> Com
linBulk b = case b of
  Bn n   -> iterate ((B:#        B):#) B !! (n - 1)
  Cn n   -> iterate ((B:#(B:#C):#B):#) C !! (n - 1)
  Sn n   -> iterate ((B:#(B:#S):#B):#) S !! (n - 1)
  x :# y -> linBulk x :# linBulk y
  _      -> b
\end{code}

But what if we wish to avoid memoziation?

== Bulk discount ==

At the cost of a logarithmic factor, we can build up the bulk combinators using
a technique analogous to repeated squaring when exponentiating.

Define the b0 and b1 combinators:

------------------------------------------------------------------------------
b0 c x y = c x       (B y y)
b1 c x y = c (B x y) (B y y)
X x y = x I
------------------------------------------------------------------------------

To compute $S_{50}$ for example, we write 50 in binary: 11010. Then we chain
together `b0` or `b1` depending on each bit as follows:

------------------------------------------------------------------------------
S_50 = b1(b1(b0(b1(b0( X ))))) I (B(BS)B)
------------------------------------------------------------------------------

We handle other bulk combinators similarly.

\begin{code}
logBulk :: Com -> Com
logBulk b = case b of
  -- B(BC)B = \cxyz.c(x(yz))
  -- B(BS)B = \cxyz.c(xz(yz))
  Bn n   -> go n K                 :# I :# B
  Cn n   -> go n (B:#K:#(C:#I:#I)) :# I :# B:#(B:#C):#B
  Sn n   -> go n (B:#K:#(C:#I:#I)) :# I :# B:#(B:#S):#B
  x :# y -> logBulk x :# logBulk y
  _      -> b
  where
  go n base = foldr (:#) base $ ([b0, b1]!!) <$> bits [] n
  bits acc 0 = acc
  bits acc n | (q, r) <- divMod n 2 = bits (r:acc) q
  b0 = B:#(C:#C:#(S:#B:#I)):#(B:#B)
  b1 = B:#(C:#C:#(S:#B:#I)):#(B:#(B:#S):#(C:#C:#B:#(B:#B:#B)))
\end{code}

== Show Me ==

\begin{code}
instance Show Com where
  show S = "S"
  show I = "I"
  show C = "C"
  show K = "K"
  show B = "B"
  show (l :# r) = (show l ++) $ case r of
    _ :# _ -> "(" ++ show r ++ ")"
    _      ->        show r
  show (Bn n) = "B_" ++ show n
  show (Cn n) = "C_" ++ show n
  show (Sn n) = "S_" ++ show n
\end{code}

For example:
------------------------------------------------------------------------------
logBulk (Sn 1234) =
B(CC(SBI))(B(BS)(CCB(BBB)))(B(CC(SBI))(BB)(B(CC(SBI))(BB)(B(CC(SBI))(B(BS)(CCB(BBB)))(B(CC(SBI))(B(BS)(CCB(BBB)))(B(CC(SBI))(BB)(B(CC(SBI))(B(BS)(CCB(BBB)))(B(CC(SBI))(BB)(B(CC(SBI))(BB)(B(CC(SBI))(B(BS)(CCB(BBB)))(B(CC(SBI))(BB)(BK(CII))))))))))))I(B(BS)B)

logBulk (Bn 1234) =
B(CC(SBI))(B(BS)(CCB(BBB)))(B(CC(SBI))(BB)(B(CC(SBI))(BB)(B(CC(SBI))(B(BS)(CCB(BBB)))(B(CC(SBI))(B(BS)(CCB(BBB)))(B(CC(SBI))(BB)(B(CC(SBI))(B(BS)(CCB(BBB)))(B(CC(SBI))(BB)(B(CC(SBI))(BB)(B(CC(SBI))(B(BS)(CCB(BBB)))(B(CC(SBI))(BB)K))))))))))IB
------------------------------------------------------------------------------

== Demo ==

\begin{code}
source :: Parsec String [String] Deb
source = term where
  term = lam <|> app
  lam = do
    orig <- getState
    vs <- between lam0 lam1 (many1 v)
    modifyState (reverse vs ++)
    t <- term
    putState orig
    pure $ iterate Lam t !! length vs
    where lam0 = str "\\" <|> str "\0955"
          lam1 = str "->" <|> str "."
  v   = many1 alphaNum <* ws
  app = foldl1' App <$> many1 ((ind =<< v) <|>
    between (str "(") (str ")") term)
  ind s = (iterate Succ Zero !!) .
    maybe (error $ s ++ " is free") id . elemIndex s <$> getState
  str = (>> ws) . string
  ws = many (oneOf " \t") >> optional (try $ string "--" >> many (noneOf "\n"))

#ifdef __HASTE__
main = withElems ["input", "lin", "log", "skiB"] $
    \[iEl, linEl, logEl, skiB] -> do
  skiB `onEvent` Click $ const $ do
    s <- getProp iEl "value"
    let v = snd . ski <$> runParser source [] "" s
    setProp linEl "value" $ either (("error" ++) . show) show v
    setProp logEl "value" $ either (("error" ++) . show) show $ logBulk <$> v
#else
main = print $ logBulk $ Sn 1234
#endif
\end{code}
