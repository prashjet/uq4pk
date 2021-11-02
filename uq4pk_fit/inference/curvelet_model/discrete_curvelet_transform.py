# This is a wrapper for PyCurvelab

import numpy as np
import pyct

class DiscreteCurveletTransform():
    def __init__(self, n1, n2, nscales, nangles, usecoarse=True):
        self.n1 = n1
        self.n2 = n2
        self.nbase = n1*n2
        self.DCT = pyct.fdct2((n1,n2), nbs=nscales, nba=nangles,
                        ac=usecoarse, vec=True, cpx=False)
        self.ncoeff = len(self.zeroCLarray())
        # finally, compute the Jacobian matrix of inverse curvelet transform
        self.PhiInv = self.inverse_transform_matrix()
        self.Phi = self.transform_matrix()

    def fwd(self, vector):
        if np.any(np.isinf(vector)):
            coeffs = np.inf * np.ones(self.ncoeff)
        else:
            image = np.reshape(vector, (self.n1, self.n2))
            coeffs = np.array(self.DCT.fwd(image))
        return coeffs

    def inv(self, coeffs):
        if np.any(np.isinf(coeffs)):
            vector = np.inf * np.ones(self.n1*self.n2)
        else:
            vector = self.DCT.inv(coeffs).flatten()
        return vector

    def zeroCLarray(self):
        zeroImage = np.zeros((self.n1, self.n2))
        zeroCoeffs = self.DCT.fwd(zeroImage)
        return zeroCoeffs

    def single_curvelet(self, i):
        """
        Returns the i-th single curvelet as an image
        :param i:
        :return:
        """
        coeffs= self.zeroCLarray()
        coeffs[i]=1.0
        curvelet = self.inv(coeffs)
        curvelet_image = np.reshape(curvelet, (self.n1, self.n2))
        return curvelet_image

    def transform_matrix(self):
        J = np.zeros((self.ncoeff,self.n1, self.n2))
        basevec = np.zeros((self.n1,self.n2))
        for i in range(self.n1):
            for j in range(self.n2):
                basevec[i,j]=1.0
                J[:,i,j] = self.DCT.fwd(basevec)
                basevec[i,j]=0.0
        J = J.reshape(J.shape[0],-1)
        return J

    def inverse_transform_matrix(self):
        # the columns of the Jacobian are simply the inverse DCT applied to corresponding basis vector
        J = np.zeros((self.nbase,self.ncoeff))
        # I know, the following is just ugly af
        # but at least, this is only done once in the whole program
        basevec = np.zeros(self.ncoeff)
        for i in range(self.ncoeff):
            basevec[i]=1.0
            J[:,i] = self.inv(basevec).flatten()
            basevec[i]=0.0
        return J

