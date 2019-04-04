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

http://okmij.org/ftp/tagless-final/ski.pdf[Oleg Kiselyov describes a
linear-time and linear-space algorithm for converting lambda terms to
combinators]:

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

\[
\begin{align}
B_n f g x_n ... x_1&=  f             &(g x_n ... x_1)  \\
C_n f g x_n ... x_1&= (f x_n ... x_1)& g  \\
S_n f g x_n ... x_1&= (f x_n ... x_1)&(g x_n ... x_1)  \\
\end{align}
\]

In particular, $B_1, C_1, S_1$ are the standard $B, C, S$ combinators. Linear
complexity is implied by:

\begin{code}
linBulk :: Com -> Com
linBulk b = case b of
  Bn n   -> iterate ((B:#        B):#) B !! (n - 1)
  Cn n   -> iterate ((B:#(B:#C):#B):#) C !! (n - 1)
  Sn n   -> iterate ((B:#(B:#S):#B):#) S !! (n - 1)
  x :# y -> linBulk x :# linBulk y
  _      -> b
\end{code}

== Bulk discount ==

How about without memoization? Observe:

\[
\begin{align}
B_{m+n} f &= B_m (B_n f)  \\
C'_{m+n} f &= C'_m (C'_n f)  \\
S'_{m+n} f &= S'_m (S'_n f)  \\
\end{align}
\]
where

\[
\begin{align}
C'_n c f g x_n ... x_1 &= c (f x_n ... x_1)&g  \\
S'_n c f g x_n ... x_1 &= c (f x_n ... x_1)&(g x_n ... x_1) \\
\end{align}
\]

Hence we can build up the bulk combinators in a manner analogous to repeated
squaring when exponentiating, resulting in a logarithmic factor in lieu of a
linear one. In short: binary, not unary.

Define the following combinators:

\[
\begin{align}
b_0 c x y &= c (B x x) y  \\
b_1 c x y &= c (B x x) (B x y)  \\
X x y &= y I  \\
\end{align}
\]

To compute $S_{50}$ for example, write 50 in binary: 11010. Then chain
together $b_0$ or $b_1$ depending on the bits:

\[
S_{50} = b_1(b_1(b_0(b_1(b_0 X)))) S'_1 I
\]

Similarly for the other bulk combinators.

\begin{code}
logBulk :: Com -> Com
logBulk b = case b of
  -- C' = \cfgx.c(fx) g   = B(BC)B
  -- S' = \cfgx.c fx (gx) = B(BS)B
  Bn n   -> go n (K:#I)         :# B              :# I
  Cn n   -> go n (K:#(C:#I:#I)) :# (B:#(B:#C):#B) :# I
  Sn n   -> go n (K:#(C:#I:#I)) :# (B:#(B:#S):#B) :# I
  x :# y -> logBulk x :# logBulk y
  _      -> b
  where
  go n base = foldr (:#) base $ ([b0, b1]!!) <$> bits [] n
  bits acc 0 = reverse acc
  bits acc n | (q, r) <- divMod n 2 = bits (r:acc) q
  b0 = C:#B:#(S:#B:#I)
  b1 = C:#(B:#S:#(B:#(B:#B):#(C:#B:#(S:#B:#I)))):#B
\end{code}

Therefore, if memoization is forbidden, we can easily transform a lambda term of length N to a combinatory logic term of length O(N log N).

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
λ> print $ logBulk $ Sn 1234
CB(SBI)(C(BS(B(BB)(CB(SBI))))B(CB(SBI)(CB(SBI)(C(BS(B(BB)(CB(SBI))))B(CB(SBI)(C(BS(B(BB)(CB(SBI))))B(C(BS(B(BB)(CB(SBI))))B(CB(SBI)(CB(SBI)(C(BS(B(BB)(CB(SBI))))B(K(CII))))))))))))(B(BS)B)I
λ> print $ logBulk $ Bn 1234
CB(SBI)(C(BS(B(BB)(CB(SBI))))B(CB(SBI)(CB(SBI)(C(BS(B(BB)(CB(SBI))))B(CB(SBI)(C(BS(B(BB)(CB(SBI))))B(C(BS(B(BB)(CB(SBI))))B(CB(SBI)(CB(SBI)(C(BS(B(BB)(CB(SBI))))B(KI)))))))))))BI
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
amain = print $ logBulk $ Sn 1234
main = do
  let
    s = "\\l.l(\\h t x.t(\\c n.c h(x c n)))(\\a.a)(\\a b.b)"
    Right out = logBulk <$> snd . ski <$> runParser source [] "" s
  print out
#endif
\end{code}
