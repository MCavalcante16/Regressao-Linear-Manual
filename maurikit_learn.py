import numpy as np

class LinearRegressionAnalyticUnivariate():
    def __init__(self):
        pass
    
    def fit(self, X, y):
        self.xMean = np.mean(X)
        self.yMean = np.mean(y)
        
        upb1 = np.sum((X - self.xMean) * (y - self.yMean))
        downb1 = np.sum((X-self.xMean) ** 2)
        self.b1 = upb1/downb1
        
        self.b0 = self.yMean - np.sum(self.b1 * self.xMean)
        
        print("Coeficientes - b0: " + str(self.b0) + " - b1: " + str(self.b1))
        
    def predict(self, X):
        return self.b1 * X + self.b0
    
    
class LinearRegressionGradientUnivariate(): 
    def __init__ (self):
        self.w0 = 0.1;
        self.w1 = 0.1;
        self.history = [];
        pass
    
    def fit (self, X, y, learning_rate=0.02, epochs=30):
        for i in range(0, epochs):
            print("Epoch: " + str(i))
            e = (np.sum(y - self.w1 * X - self.w0))/X.shape[0]
            ex = (np.sum((y - self.w1 * X - self.w0) * X))/X.shape[0]
            self.w0 = self.w0 + learning_rate * e
            self.w1 = self.w1 + learning_rate * ex
            self.history.append([self.w0, self.w1])
            
        print("Coeficientes - b0: " + str(self.w0) + " - b1: " + str(self.w1))
            
    def predict (self, X):
        return self.w0 + self.w1 * X
    
class PolinomialRegression():
    def __init__(self):
        pass
    
    def fit(self, X, y, grau = 2):
        #Redimensiona 
        X1 = X.reshape(X.shape[0], 1)
        
        #Gera novas colunas com grau elevado - TODO - Descobrir pq ta gerando uma coluna a mais
        self.grau = grau
        X = X1
        for i in range(1, grau):
            Xn = X1**(i+1)
            X = np.hstack((X,Xn))
       
        #Inclui bias
        bias = np.ones((X.shape[0], 1))
        X = np.hstack((bias,X))
        
        #Calcula bs
        xtx = np.matmul(np.transpose(X),X)
        inv = np.linalg.pinv(xtx)
        xxy = np.matmul(np.transpose(X), y)
        self.b = np.matmul(inv, xxy)
        print("Coeficientes: " + str(self.b))
        
    def predict(self, X):
        #Gerando X^n
        X1 = X.reshape(X.shape[0], 1)
        X = X1
        for i in range(1, self.grau):
            Xn = X1**(i+1)
            X = np.hstack((X,Xn))
        
        #Aplicando bs
        result = np.array([])
        for i in range(0, X.shape[0]):
            result = np.append(result, self.b[0] + np.sum(self.b[1:] * X[i]))
        return result
    
        
class LinearRegression():
    def __init__(self, method='analytic'):
        self.method = method
    
    def fit (self, X, y, epochs=30, learning_rate=0.02, regularizar=False, regularizacao=1):
        if (self.method == 'analytic'):
            bias = np.ones((X.shape[0], 1))
            X = np.hstack((bias,X))
            xtx = np.matmul(np.transpose(X),X)
            inv = np.linalg.pinv(xtx)
            xxy = np.matmul(np.transpose(X), y)
            self.b = np.matmul(inv, xxy)
            print("Coeficientes: " + str(self.b))
        
        elif (self.method == 'gradient'): 
            #Lista com erros
            self.mses = np.array([])
            
            #Bias
            bias = np.ones((X.shape[0], 1))
            X = np.hstack((bias,X))
            self.w = np.ones(X.shape[1])
            
            for i in range(0, epochs):
                #print("Epoch: " + str(i))
                
                #Calculo do Somatório(ei * xi) ou (ei * xij - reg/n * wj)
                exi = 0     
                for j in range(0, X.shape[0]):
                    exi += (y[j] - ((self.w * X[j]) * (-1))) * X[j]
                
                exi_n = (exi/X.shape[0])
                
                if (regularizar):
                    self.w = self.w + (learning_rate * (exi_n - regularizacao * np.transpose(self.w) * self.w) ) 
                
                else:
                    self.w = self.w + (learning_rate * exi_n)
                
                #MSE's
                previous_result = np.array([])
                for i in range(0, X.shape[0]):
                    previous_result = np.append(previous_result, np.sum(self.w * X[i]))
                self.mses = np.append(self.mses, MSE(y, previous_result))
               

            print("Coeficientes: " + str(self.w))
            
        elif (self.method == 'gradient-stochastic'):
            #Lista com erros
            self.mses = np.array([])
            
            # Bias
            bias = np.ones((X.shape[0], 1))
            X = np.hstack((bias,X))
            self.w = np.ones(X.shape[1])
            
            for i in range(0, epochs):
                #print("Epoch: " + str(i))
                for j in range(0, X.shape[0]):
                    ex = (y[j] - ((self.w * X[j]) * (-1))) * X[j]
                    self.w = self.w + learning_rate * ex
                
                #MSE's
                previous_result = np.array([])
                for i in range(0, X.shape[0]):
                    previous_result = np.append(previous_result, np.sum(self.w * X[i]))
                self.mses = np.append(self.mses, MSE(y, previous_result))
                    
            print("Coeficientes: " + str(self.w))
            
    def predict(self, X):
        if (self.method == 'analytic'):
            result = np.array([])
            for i in range(0, X.shape[0]):
                result = np.append(result, self.b[0] + np.sum(self.b[1:] * X[i]))
            return result
        
        elif (self.method == 'gradient' or self.method == 'gradient-stochastic'):
            result = np.array([])
            for i in range(0, X.shape[0]):
                result = np.append(result, self.w[0] + np.sum(self.w[1:] * X[i]))
            return result

        
def MSE (y_true, y_predict): 
    return (np.sum((y_true - y_predict) ** 2))/y_true.shape[0]

def RSS (y_true, y_predict):
    return np.sum((y_true - y_predict) ** 2)

def TSS (y):
    mean = np.mean(y)
    return np.sum((y - mean) ** 2)

def R2 (y_true, y_predict):
    return 1 - (RSS(y_true, y_predict)/TSS(y_true))
        

                                                 
        
        
        
