def scale_add(vector, alpha, vector2):
    """
    Multiplies `alpha` by every value in `vector2`, then
    adds that result with `vector`
    """
    v3 = [alpha * item for item in vector2]
    return [x + y for x, y in zip(vector, v3)]

def is_converged(vector1, vector2, diff_func, threshold=1e-8):
    """
    Returns True if the difference of `vector1` and `vector2`
    evaluated by the `diff_func.f` is less than the threshold;
    """
    return abs(diff_func.f(vector1) - diff_func.f(vector2)) < threshold
