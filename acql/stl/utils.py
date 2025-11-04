# see Data.Tree foldTree from haskell

def fold_tree(f, expression):
    """
    f :: (a -> [b] -> b)
    expression :: Tree a
    out :: b
    """
    bs = map(lambda x: fold_tree(f, x), expression.children)
    return f(expression, bs)

def fold_spot_formula(f, formula):
    """
    f :: (a -> [b] -> b)
    expression :: SpotFormula
    out :: b
    """
    bs = map(lambda x: fold_spot_formula(f, x), formula)
    return f(formula, bs)
