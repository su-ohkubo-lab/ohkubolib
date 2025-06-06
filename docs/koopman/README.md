# ohkubolib.koopman

## Dictionary (抽象クラス)
新しい辞書クラスを定義する際に継承するための抽象クラス

例)
```python
from ohkubolib.koopman import Dictionary

class NewDictionary(Dictionary):
    # コンストラクタ
    def __init__(self) -> None:
        pass

    # 辞書
    def __call__(self, X: np.ndarray) -> np.ndarray:
        pass

    # 辞書の1階微分
    def diff(self, X: np.ndarray) -> np.ndarray:
        pass

    # 辞書の2階微分
    def ddiff(self, X: np.ndarray) -> np.ndarray:
        pass

```

## Monomial
単項式辞書のクラス

e.g.)
```python
import numpy as np
from ohkubolib.koopman import Monomial

X = np.rand(2, 100)
Y = np.rand(2, 100)

# 最大次数 3 の単項式辞書をインスタンス化
monomial = Monomial(3)
PsiX = monomial(X)
dPsiX = monomial.diff(X)
ddPsiX = monomial.ddiff(X)

# 2 変数の最大次数 3 の単項式の冪乗を計算するための numpy 配列を取得
dim = 2
order = 3
powers = Monomial.make_powers(dim, order)
```

> [!TIP]
> ```make_powers``` はスタティックメソッドであるため, インスタンス化せずに呼び出せる.

## EDMD
EDMD のクラス

e.g.) x ベクトルと y ベクトルに対する辞書が同じ場合
```python
import numpy as np
from ohkubolib.koopman import Monomial, EDMD

X = np.rand(2, 100)
Y = np.rand(2, 100)
dictionary = Monomial(3)

# EDMDをインスタンス化
edmd = EDMD(dictionary)

# Koopman行列 K を計算
K = edmd(X, Y)

# Koopman行列 K を保存
edmd.save('./K.csv')
```

e.g.) x ベクトルと y ベクトルに対する辞書が異なる場合
```python
import numpy as np
from ohkubolib.koopman import Monomial, EDMD

X = np.rand(2, 100)
Y = np.rand(2, 100)
dictionary_X = Monomial(2)
dictionary_Y = Monomial(3)

# EDMDをインスタンス化
edmd = EDMD(dictionary_X=dictionary_X, dictionary_Y=dictionary_Y)

# Koopman行列 K を計算
K = edmd(X, Y)

# Koopman行列 K を保存
edmd.save('./K.csv')
```

## Online EDMD
Online EDMD のクラス

今後実装予定.

## Lifting
Koopman Lifting のクラス (EDMD クラスを継承)

e.g.)
```python
import numpy as np
from ohkubolib.koopman import Monomial, Lifting

X = np.rand(2, 100)
Y = np.rand(2, 100)
dictionary = Monomial(3)

# Liftingをインスタンス化
lifting = Lifting(dictionary)

# Koopman行列 K & Koopman 生成子行列 L を計算
L = lifting(X, Y)

# Koopman行列 K を保存
lifting.save('./K.csv', data='K')

# Koopman生成子行列 L を保存
lifting.save('./L.csv')
```

## gEDMD
gEDMD のクラス

e.g.) 通常のgEDMD
```python
import numpy as np
from ohkubolib.koopman import Monomial, gEDMD

X = np.rand(2, 100)
Y = np.rand(2, 100)
dictionary = Monomial(3)

# gEDMDをインスタンス化
gedmd = gEDMD(dictionary)

# Koopman生成子行列 L を計算
L = gedmd(X, Y)

# Koopman生成子行列 L を保存
gedmd.save('./L.csv')
```

e.g.) Lasso正則化ありのgEDMD
```python
import numpy as np
from ohkubolib.koopman import Monomial, gEDMD

X = np.rand(2, 100)
Y = np.rand(2, 100)
dictionary = Monomial(3)

# gEDMDをインスタンス化
gedmd = gEDMD(dictionary, mode='lasso', alpha=1.0, iterations=1000)

# Koopman生成子行列 L を計算
L = gedmd(X, Y)

# Koopman生成子行列 L を保存
gedmd.save('./L.csv')
```

e.g.) Ridge正則化ありのgEDMD
```python
import numpy as np
from ohkubolib.koopman import Monomial, gEDMD

X = np.rand(2, 100)
Y = np.rand(2, 100)
dictionary = Monomial(3)

# gEDMDをインスタンス化
gedmd = gEDMD(dictionary, mode='ridge', alpha=1.0, iterations=1000)

# Koopman生成子行列 L を計算
L = gedmd(X, Y)

# Koopman生成子行列 L を保存
gedmd.save('./L.csv')
```