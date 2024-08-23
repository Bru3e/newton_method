def f(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def gradient(f, x, h=1e-5):
    n = len(x)
    grad = [0.0] * n
    for i in range(n):
        x_plus = x[:]
        x_minus = x[:]
        x_plus[i] += h
        x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad

def hessian(f, x, h=1e-5):
    n = len(x)
    hess = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            x_ijpp = x[:]
            x_ijmm = x[:]
            x_ijpm = x[:]
            x_ijmp = x[:]
            
            x_ijpp[i] += h
            x_ijpp[j] += h
            
            x_ijmm[i] -= h
            x_ijmm[j] -= h
            
            x_ijpm[i] += h
            x_ijpm[j] -= h
            
            x_ijmp[i] -= h
            x_ijmp[j] += h
            
            hess[i][j] = (f(x_ijpp) - f(x_ijpm) - f(x_ijmp) + f(x_ijmm)) / (4 * h**2)
    return hess

def multivariate_newton_method(f, x0, tol=1e-6, max_iter=100):
    x = x0
    for _ in range(max_iter):
        grad = gradient(f, x)
        hess = hessian(f, x)
        hess_inv = [[0] * len(x) for _ in range(len(x))]
        for i in range(len(x)):
            hess_inv[i][i] = 1 / hess[i][i] if hess[i][i] != 0 else 0

        x_new = [x[i] - sum(hess_inv[i][j] * grad[j] for j in range(len(x))) for i in range(len(x))]
        
        if all(abs(x_new[i] - x[i]) < tol for i in range(len(x))):
            return x_new
        x = x_new
    
    return x

# Example usage
x0 = [-1.2, 1]
result = multivariate_newton_method(f, x0)
print("Minimum found at:", result)
