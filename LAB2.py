import numpy as np
from time import time
import matplotlib.pyplot as plt
from math import *
import pandas as pd
import sklearn
import sklearn.preprocessing
import sklearn.decomposition
from sklearn import linear_model
from mpl_toolkits.mplot3d import Axes3D
from sklearn.pipeline import Pipeline

class EX1:
    def __init__(self, n, p, verbose=False):
        self.verbose = verbose
        self.n = n
        self.p = p
        # CONSTRUCT LAMBDA = 1E-5
        self._lambda = 1.0e-05
        # CONSTRUCT X: N*P ~ N (0,5)
        self._matX = np.random.normal(0, 5, [self.n, self.p])
        # CONSTRUCT Y: N*1 ~ U (-1,1)
        self._vecY = np.random.uniform(-1, 1, [self.n, 1])
        if self.verbose:
            self.print_self()
        return

    def print_self(self):
        print(self._lambda)
        print(self._vecY)
        print(self._matX)

    def check_kernel_trick(self):
        # (XTX + lambda * I(p)) * XT * Y = XT * (XXT + lambda * I(n)) * y
        mat_left = np.dot((np.dot(self._matX.T, self._matX) + self._lambda * np.identity(self.p)), self._matX.T)
        mat_right = np.dot(self._matX.T, np.dot(self._matX, self._matX.T) + self._lambda * np.identity(self.n))
        return np.allclose(mat_left, mat_right)

    def calculate_left(self):
        return np.dot(np.dot(self._matX.T,
                             np.linalg.inv(np.dot(self._matX, self._matX.T) + self._lambda * np.identity(self.n))
                             ), self._vecY)

    def calculate_right(self):
        return np.dot(np.dot(
            np.linalg.inv(np.dot(self._matX.T, self._matX) + self._lambda * np.identity(self.p)),
            self._matX.T), self._vecY)


# ex1_test = EX1(100, 2000, False)

# start = time()

# for i in range(10):
#    left = ex1_test.calculate_left()
#
# print(time() - start)
#
# start = time()
#
# for i in range(10):
#     right = ex1_test.calculate_right()
#
# print(time() - start)
#
# print(np.allclose(left, right, atol = 1e-5, rtol = 0.01))

def EX1_Test_Perform():
    perfA = []
    perfB = []
    testR = []
    n = [5, 10, 20, 50, 100, 200, 500, 1000, 2000]
    p = list(map(lambda x: int(10000 / x), n))
    for i in range(len(n)):
        print(n[i], p[i])
        test_obj = EX1(n[i], p[i])
        start = time()
        for j in range(10):
            left = test_obj.calculate_left()
        perfA.append(time() - start)

        start = time()
        for j in range(10):
            right = test_obj.calculate_right()
        perfB.append(time() - start)

        testR.append(np.allclose(left, right, atol=1e-5, rtol=0.01))
    plt.subplot(1, 2, 1)
    plt.plot(perfA)
    plt.subplot(1, 2, 2)
    plt.plot(perfB)
    plt.show()
    return perfA, perfB


#EX1_Test_Perform()

class EX2:
    def __init__(self):
        pass

    def generate_random_matrix(n, p, distribution):
        # DISTRIBUTIONS
        # 1. Binominal (8, 0.5) - 4
        # 2. Uniform (-sr(6), sr(6))
        # 3. Laplace (miu = 0, b = 1)
        if 'Binomial' == distribution:
            return np.random.binomial(8, 0.5, [n, p]) - 4
        elif 'Uniform' == distribution:
            return np.random.uniform(-sqrt(6), sqrt(6), [n, p])
        elif 'Laplace' == distribution:
            return np.random.laplace(0, 1, [n, p])
        else:
            return None

    def test_generate_and_plot(n, p, distribution):
        mat = EX2.generate_random_matrix(n, p, distribution)
        # print(np.reshape(mat, n*p))
        plt.violinplot(np.reshape(mat, n * p))
        plt.show()

    def test_generate_and_plot_spectrum(n, p, distribution):
        mat = EX2.generate_random_matrix(n, p, distribution)
        mat = np.dot(mat, mat.T) / n
        eigV = np.linalg.eigvals(mat)
        plt.violinplot(eigV)
        plt.show()


# EX2.test_generate_and_plot_spectrum(200,200, 'Binomial')
# EX2.test_generate_and_plot_spectrum(200,500, 'Binomial')
# EX2.test_generate_and_plot(200,1000, 'Binomial')
# EX2.test_generate_and_plot_spectrum(200,2000, 'Binomial')
# EX2.test_generate_and_plot_spectrum(200,5000, 'Binomial')

class PC2_EX3:

    def __init__(self, n, p, niter):
        # data members
        self._vecu = None
        self._vecv = None
        self.iter = 0
        self._matX = None
        self._matArchX = None
        self.p = p
        self.n = n
        self._residuenormu = None
        self._residuenormv = None

        self.set_niter(niter)
        self.reinit_matX()
        self.reinit_vecu()
        self.reinit_vecv()

    # deprecated constructor
    # matX, vecu, vecv should be generated inside the object

    # def __init__(self, u, v, niter):
    #     # data members
    #     self._vecu = None
    #     self._vecv = None
    #     self.iter = 0
    #     self._matX = None
    #     self._matArchX = None
    #     self.p = 0
    #     self.n = 0
    #     self._residuenormu = None
    #     self._residuenormv = None
    #
    #     self.set_niter(niter)
    #     self.set_vecv(v)
    #     self.set_vecu(u)
    #     # Construct X
    #     self.reinit_matX()
    #     return

    '''
    INTERFACE
    '''

    def print_self(self):
        '''
        :summary: for debugging only
        :return: VOID
        '''
        print("Object Type: ", type(self))
        print("Ptr: ", id(self))
        print("N * P = ", self.n, " * ", self.p)
        print("NrIter = ", self.iter)

    def get_matX(self):
        return self._matX.copy()

    def get_vecu(self):
        return self._vecu

    def get_vecv(self):
        return self._vecv

    def get_residu(self):
        return self._residuenormu

    def get_residv(self):
        return self._residuenormv

    def reinit_matX(self):
        self._matX = np.random.normal(0, 5, [self.n, self.p])
        return

    def reinit_vecu(self):
        self._vecu = np.random.normal(0, 200, [self.n, 1])
        return

    def reinit_vecv(self):
        self._vecv = np.random.normal(0, 200, [self.p, 1])
        return

    def set_vecv(self, v):
        self._vecv = v
        self.p = len(self._vecv)

    def set_vecu(self, u):
        self._vecu = u
        self.n = len(self._vecu)

    def set_niter(self, niter):
        self.iter = niter

    '''
    IMPLEMENTATION
    '''

    def power_method_a(self):
        '''
        :summary: PC2_EX3_6
        :Members modified:

        _matX  to be used as ReadOnly
        _residuenormu to be overwritten
        _residuenormv to be overwritten
        _vecu / _vecv to be modified

        :return: VOID
        '''
        self._residuenormu = np.ndarray((self.iter,), float)
        self._residuenormv = np.ndarray((self.iter,), float)
        for i in range(self.iter):
            legacy_u = self._vecu
            legacy_v = self._vecv

            self._vecu = np.dot(self._matX, self._vecv)
            self._vecv = np.dot(self._matX.T, self._vecu)

            self._vecu /= np.linalg.norm(self._vecu, 2)
            self._vecv /= np.linalg.norm(self._vecv, 2)

            legacy_u -= self._vecu
            legacy_v -= self._vecv

            self._residuenormu[i] = np.linalg.norm(legacy_u, 2)
            self._residuenormv[i] = np.linalg.norm(legacy_v, 2)

        return

    def power_method_b(self, matX, niter=1000):
        '''
        :summary: decoupled static function
        :algorithm:
        1. find dominant eigenvector of XTX by power method
        2. find associate eigenvalue by Rayleigh quotient
        3. sqrt(eigenvalue of XTX) = corresponding singular value of X
        :return:
        dominant singular value of matX
        '''

        matXTX = np.dot(matX.T, matX)
        v = np.random.normal(0, 200, [matXTX.shape[0]])
        for i in range(niter):
            v = np.dot(matXTX, v)
            v /= np.linalg.norm(v, 2)
        return sqrt((np.dot(np.dot(v.T, matXTX), v)) / (np.dot(v.T, v)))

    def get_second_sv(self):
        '''
        :members modified:
        vecv / vecu will be overwritten by a new pair of singular vectors

        # 1. get first s-value by power method B
        # 2. get s-vectors u,hv associated to 1st s-value by power method A
        # 3. deduct component of u * s-value * vh from matX
        # 4. re-apply power method B to find the largest s-value,
        #   and this is exactly what we need
        '''
        # download matX from the class object,
        # I do not want to modify it during this function
        matX = self._matX.copy()

        # get largest sv
        sv0 = self.power_method_b(matX)

        # get associated s-vector
        self.reinit_vecu()
        self.reinit_vecv()
        self.power_method_a()

        # construct component and deduct it from matX
        smat = np.zeros((self.n, self.p))
        smat[0, 0] = sv0
        umat = np.zeros((self.n, self.n))
        vhmat = np.zeros((self.p, self.p))
        umat[:, 0] = self._vecu[:, 0]
        vhmat[0] = self._vecv[:, 0]

        matX = matX - np.dot(np.dot(umat, smat), vhmat)

        # get second largest sv
        return self.power_method_b(matX)

    def reduced_svd(self):
        '''
        :summary: to simulate the function of np.linalg.svd(X, FullMat = False)
        :remarks: SVD
                            |SV[0] .........|
                            |..SV[1] .......|
                U(N*P)   *  |....SV[2]......|   * VH (P*P)
                            |......SV[3]....|
                            |........SV[P-1]|
                A good approximation to input matrix X
        :algorithm:
        loop for p times:
            get largest s-value by power method B, write to _svalues[p]
            get associated u, vh by power method A, write to _smatu[:,p] and _smatvh[p,:]
            duduct U * diag(SValues) * VH from matX
        :return:
        1. matrix U (n*p)
        2. S-Values SV (p, descending order)
        3. matrix VH (p*p)
        '''

    def power_method_to_find_second_largest_singular_value(self):
        uh, svalue, vh = np.linalg.svd(self._matX, False)
        print("SV by numpy: ", svalue)
        smat = np.zeros(self._matX.shape)
        smat[0, 0] = svalue[0]
        umat = np.zeros((self.n, self.n))
        vhmat = np.zeros((self.p, self.p))
        umat[:, 0] = self._vecu[:, 0]
        vhmat[0] = self._vecv[:, 0]
        print(np.dot(np.dot(umat, smat), vhmat))
        tmatX = self._matX - np.dot(np.dot(umat, smat), vhmat)
        uh, svalue2, vh = np.linalg.svd(tmatX, False)
        print("SV by numpy: ", svalue2)

    def reset_vector_and_execute(self, u, v):
        assert len(v) == self.p
        assert len(u) == self.n
        self.set_vecv(v)
        self.set_vecu(u)
        self.execute_power_method()
        print(np.reshape(self._vecu, [self.n]))
        print(np.reshape(self._vecv, [self.p]))

    def power_method_to_find_largest_singular_value(self):
        matXTX = np.dot(self._matX.T, self._matX)
        v = np.random.normal(0, 200, [self.p])
        for i in range(self.iter):
            v = np.dot(matXTX, v)
            v /= np.linalg.norm(v, 2)

        return sqrt((np.dot(np.dot(v.T, matXTX), v)) / (np.dot(v.T, v)))


# PC2_EX3_Q7
# Viz the convergence and check whether u,v is dominant singular vectors
def PC2_EX3_7():
    # init
    ex3_obj = PC2_EX3(100, 100, 1000)
    matX = ex3_obj.get_matX()

    # get u, vh associated to dominant singular value
    ex3_obj.power_method_a()
    u0 = ex3_obj.get_vecu()
    v0 = ex3_obj.get_vecv()
    # get norm of delta u / delta v
    residu = ex3_obj.get_residu()
    residv = ex3_obj.get_residv()
    # viz
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.plot(residu, label='Residue Norm U')
    ax1.plot(residv, label='Residue Norm V')
    plt.legend(loc='upper right')
    plt.yscale('log')
    plt.title('Residue Norm')
    plt.show()

    # SVD by numpy
    uh, svalue, vh = np.linalg.svd(matX, False)
    print('Whether U corresponds to singular vector ? ',
          ((np.allclose(uh[:, 0], np.reshape(u0, [u0.shape[0]]), 0.01, 1e-4)
            or np.allclose(-uh[:, 0], np.reshape(u0, [u0.shape[0]]), 0.01, 1e-4))
          ))
    print('Whether V corresponds to singular vector ? ',
          ((np.allclose(vh[0], np.reshape(v0, [v0.shape[0]]), 0.01, 1e-4)
            or np.allclose(-vh[0], np.reshape(v0, [v0.shape[0]]), 0.01, 1e-4))
          ))

    return


# PC2_EX3_Q8
# two convergences: u, vh & -u, -vh
def PC2_EX3_8():
    # init
    ex3_obj = PC2_EX3(100, 100, 1000)

    # get u, vh associated to dominant singular value
    ex3_obj.power_method_a()
    u0 = ex3_obj.get_vecu()
    v0 = ex3_obj.get_vecv()
    # print("VecU0: ", u0.reshape)
    # print("VecH0: ", v0)

    # add a small disturbance to -u0 / -v0
    ex3_obj.set_vecu(np.random.normal(0, 1, u0.shape) - u0)
    ex3_obj.set_vecv(np.random.normal(0, 1, v0.shape) - v0)

    # recalculate and get new u, vh pair
    ex3_obj.power_method_a()
    u1 = ex3_obj.get_vecu()
    v1 = ex3_obj.get_vecv()
    # print("VecU1: ", u1)
    # print("VecH1: ", v1)
    print("Whether U0 = -U1 ? ",
          np.allclose(u0, -u1, 0.001, 1e-5))
    print("Whether V0 = -V1 ? ",
          np.allclose(v0, -v1, 0.001, 1e-5))

    return


# PC2_EX3_Q9_Q10
# Appmx first and second largest singular value
def PC2_EX3_9():
    # init
    ex3_obj = PC2_EX3(100, 100, 1000)
    matX = ex3_obj.get_matX()

    # get largest singular value
    print('SV0: ', ex3_obj.power_method_b(matX))

    # get second largest singular value
    print('SV1: ', ex3_obj.get_second_sv())

    # calculate S-VALUE by NUMPY SVD
    _, sv, _ = np.linalg.svd(matX, False)

    print('SV0 + SV1 (by numpy): ', sv[0], ' ', sv[1])

    return


class PC2_EX4:

    def __init__(self):
        # fields = ['England', 'Wales', 'Scotland', 'N Ireland']
        self.df = pd.read_csv('./data/defra_consumption.csv', sep=';',
                              index_col=0, header=0)
        #print(self.df)
        self.anno = self.df.columns
        #print(self.anno)
        return

    def standardize(self):
        self.df = sklearn.preprocessing.scale(self.df, 1, True, True, False)
        self.df = self.df.T
        #print(self.df)
        #print(self.df.shape)

    def PCA_2(self):
        self.standardize()
        pca = sklearn.decomposition.PCA(n_components=2)
        pca.fit(self.df)
        df1 = pca.fit_transform(self.df)
        # print(pca.explained_variance_ratio_)
        # print(df1)
        fig = plt.figure()
        ax = fig.subplots()
        plt.scatter(df1[:,0], df1[:,1])
        for i, txt in enumerate(self.anno):
            ax.annotate(txt, (df1[i,0], df1[i,1]))
        plt.show()

    def PCA_3(self):
        self.standardize()
        pca = sklearn.decomposition.PCA(n_components=3)
        pca.fit(self.df)
        df3 = pca.fit_transform(self.df)
        # print(pca.explained_variance_ratio_)
        # print(df3)
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(df3[:, 0], df3[:, 1], df3[:, 2])
        for i, txt in enumerate(self.anno):
            pass
            ax.text(df3[i, 0], df3[i, 1], df3[i, 2], txt, None)
            # ax.annotate(txt, (df3[i,0], df3[i,1], df3[i,2]))
        # plt.scatter(df1, np.zeros(df1.shape[0]))
        plt.show()

    def diag_xtx(self):
        self.standardize()
        xtx = np.dot(self.df.T, self.df)
        print(xtx.shape)

        #XTX is symmetric
        xtx_evalue, xtx_evector = np.linalg.eigh(xtx)
        # Ascending order
        # print(xtx_evalue)
        # print(xtx_evector)
        # Span 2 most dominant eigenvectors
        span2d = xtx_evector[:, -2: ]
        # Span 3 most dominant eigenvectors
        span3d = xtx_evector[:, -3: ]
        print(span2d)
        print("span2d orthogonal?")
        print("Inner product: ", np.inner(span2d[:,0], span2d[:,1]))

        shapeV = xtx_evector.shape[1]
        test_x = np.zeros((shapeV, shapeV))
        for i in range(shapeV):
            for j in range(shapeV):
                test_x[i, j] = np.inner(xtx_evector[i, :], xtx_evector[j, :])
        print(test_x)

        # Project X (4*17) onto span2d(17 * 2)
        proj2d = np.dot(self.df, span2d)
        print(proj2d.shape)
        fig = plt.figure()
        ax = fig.subplots()
        plt.scatter(proj2d[:, 0], proj2d[:, 1])
        for i, txt in enumerate(self.anno):
            ax.annotate(txt, (proj2d[i, 0], proj2d[i, 1]))
        plt.show()
        return

    def svd_x(self):
        self.standardize()
        print(self.df.shape)
        #uf, sf, vf = np.linalg.svd(self.df)
        #print("U ", uf.shape, " S ", sf.shape, " V ", vf.shape)

        ur, sr, vr = np.linalg.svd(self.df, False)
        print("U ", ur.shape, " S ", sr.shape, " V ", vr.shape)

        #print(vf)
        print(sr)

        print("Orthogonal? ")
        shapeV = vr.shape[0]
        test_x = np.zeros((shapeV,shapeV))
        for i in range(shapeV):
            for j in range(shapeV):
                test_x[i,j] = np.inner(vr[i,:], vr[j,:])
        print(test_x)

        #print("Inner prod: ", np.inner(vr[0, :], vr[1, :]))
        #print("NORM:", np.linalg.norm(vr[0, :]))
        #print("NORM:", np.linalg.norm(vr[1, :]))

#ex4_obj = PC2_EX4()
#ex4_obj.standardize()
#ex4_obj.diag_xtx()


class PC2_EX5:

    def __init__(self):
        cols = ['mpg'
            , 'cylinders'
            , 'displacement'
            , 'horsepower'
            , 'weight'
            , 'acceleration'
            , 'model year'
            , 'origin'
            , 'car name']
        self.df = pd.read_csv('./data/auto-mpg.data-original',
                              sep=r"\s+", names=cols)
        self.df = self.df.dropna()

        print('Nr missing data: ',
              sum((pd.isna(self.df['mpg']).astype(int))))
        print('Nr missing data: ',
              sum((pd.isna(self.df['cylinders']).astype(int))))
        print('Nr missing data: ',
              sum((pd.isna(self.df['displacement']).astype(int))))
        print('Nr missing data: ',
              sum((pd.isna(self.df['horsepower']).astype(int))))
        print('Nr missing data: ',
              sum((pd.isna(self.df['weight']).astype(int))))
        print('Nr missing data: ',
              sum((pd.isna(self.df['acceleration']).astype(int))))
        print('Nr missing data: ',
              sum((pd.isna(self.df['model year']).astype(int))))
        print('Nr missing data: ',
              sum((pd.isna(self.df['origin']).astype(int))))
        print('Nr missing data: ',
              sum((pd.isna(self.df['car name']).astype(int))))
        print('Nr data: ',
              self.df.shape[0])

        print(self.df)

    def naive_model(self):
        self.df = self.df.drop(['origin', 'car name'], axis=1)
        #self.df = self.df.set_index('car name')
        #print(self.df)
        regr = linear_model.LinearRegression(fit_intercept=True)
        self.pred_y = self.df['mpg'][0:9]
        self.pred_x = self.df[[
            'cylinders'
            , 'displacement'
            , 'horsepower'
            , 'weight'
            , 'acceleration'
            , 'model year']][0:9]
        print(self.pred_y)
        print(self.pred_x)
        regr.fit(self.pred_x, self.pred_y)

        print([
            'cylinders'
            , 'displacement'
            , 'horsepower'
            , 'weight'
            , 'acceleration'
            , 'model year'])
        print('COEF:')
        print(regr.coef_)
        print('INTERCEPT:')
        print(regr.intercept_)

    def improved_model(self):
        # Drop two qualitative variables
        self.df = self.df.drop(['origin', 'car name'], axis=1)
        # Do standardize
        scaler = sklearn.preprocessing.StandardScaler()
        self.scaled_df = scaler.fit_transform(self.df)
        print([ 'mpg'
                , 'cylinders'
                , 'displacement'
                , 'horsepower'
                , 'weight'
                , 'acceleration'
                , 'model year'])
        print(scaler.mean_)

        # self.df = self.df.set_index('car name')
        #print(self.ndarray)
        #print(np.mean(self.ndarray[:,0]))
        regr = linear_model.LinearRegression(fit_intercept=True)
        self.train_y = self.df['mpg']
        self.train_x = self.scaled_df[:, 1:]
        #
        print(self.train_y.shape)
        print(self.train_x.shape)
        regr.fit(self.train_x, self.train_y)
        #
        print([
            'cylinders'
            , 'displacement'
            , 'horsepower'
            , 'weight'
            , 'acceleration'
            , 'model year'])
        print('COEF:')
        print(regr.coef_)
        print('INTERCEPT:')
        print(regr.intercept_)

        # predicted y
        self.pred_y = regr.predict(self.train_x)
        self.residue_y = self.pred_y - self.train_y

        # VIZ
        fig = plt.figure()
        ax = fig.subplots()
        ax.scatter(np.arange(0, self.train_y.shape[0]) , self.train_y)
        ax.scatter(np.arange(0, self.train_y.shape[0]) , self.pred_y)
        plt.show()

        # CALCULATE RESIDUAL NORM
        mean_obsv_y = scaler.mean_[0]
        print(mean_obsv_y)
        # /Y
        bar_y = mean_obsv_y * np.ones(self.train_y.shape)
        #print("BAR_Y", bar_y)
        # NORM OF Y - /Y * I
        norm_std_obsv_y = np.linalg.norm(self.train_y - bar_y, 2)
        print('NORM OBSV Y: ', norm_std_obsv_y)
        # NORM OF RESIDUE R
        norm_r = np.linalg.norm(self.residue_y, 2)
        print('NORM R: ', norm_r)
        # NORM OF Y - Estm Y * I
        norm_std_estm_y = np.linalg.norm(self.pred_y - bar_y, 2)
        print('NORM ESTM Y: ', norm_std_estm_y)
        print("Is norm Y^2= norm R^2 + norm /Y^2  ? ",
              np.isclose(norm_std_obsv_y*norm_std_obsv_y,
                         norm_r*norm_r + norm_std_estm_y*norm_std_estm_y))

    def pipeline_model(self):
        # Drop two qualitative variables
        self.df = self.df.drop(['origin', 'car name'], axis=1)
        self.train_y = self.df['mpg']
        self.train_x = self.df.drop(['mpg'], axis=1)
        print(self.train_x)

        scaler = sklearn.preprocessing.StandardScaler()
        regr = linear_model.LinearRegression(fit_intercept=True)
        pipeline = Pipeline([('scaler', scaler), ('regressor', regr)])
        pipeline.fit(self.train_x, self.train_y)

        test_x = np.ndarray((1,6))
        test_x[0,:] = [6, 225, 100, 3233, 15.4, self.train_x['model year'].mean()]
        print(test_x)
        pred_y = pipeline.predict(test_x)
        print("pred_MPG", pred_y)

        return
#
#ex5 = PC2_EX5()
#ex5.improved_model()
# ex4_obj = PC2_EX4()
# ex4_obj.standardize()
# ex4_obj.PCA_3()
# print(ex4_obj.df.shape)
# u = np.random.normal(0,200,[100,1])
# v = np.random.normal(0,200,[50,1])
# ex3 = EX3(u,v,1000)
# v, u, trv, tru = ex3.execute_power_method()
# #print(ex3.check_eigen())
# ex3.power_method_to_find_second_largest_singular_value()
#
# # for i in range(10):
# #     print(i)
# #     u = np.random.normal(0,200,[100,1])
# #     v = np.random.normal(0,200,[50,1])
# #
# #     ex3.reset_vector_and_execute(u,v)
# # plt.subplot(1,2,1)
# # plt.plot(trv)
# # plt.title('L2 Norm of Residual vector V')
# # plt.yscale('log')
# # plt.subplot(1,2,2)
# # plt.plot(tru)
# # plt.title('L2 Norm of Residual vector U')
# # plt.yscale('log')
# # plt.show()
# #
# # print(v)
# # print(u)
# # print(np.allclose(u,v,rtol = 0.01, atol= 1e-05))
