import numpy as np

class Rbfcm(object):
    """Radial Basis function collocation method for 1D shallow water equations

    Parameters
    -------------
    mesh: object
        Mesh object which holds mesh properties.
        --locations, 1d array
        --nodeNo, int
        --minR, minimum idstance between any two points, float
    rbf: string
        Radial basis function identifier.
    boundaries:
        Boundary condition identifier.
    shapeParameter

    Attributes
    -------------
    f,fx,fy,fxx,fyy: 2d-array
        Radial basis function coefficients.
    system: 2d-array
        System matrix.
    rhs: 1d-array
        Right hand side of the equation, load vector.
    soln: 1d-array
        Solution of the differential equation.
    """

    def __init__(self, inputFilePath, qIni, hIni, bWidth, mannings, slope):

        # with open(inputFilePath, "r") as file:
        #     lines = file.readlines()


        self.locations = np.loadtxt(inputFilePath)
        N = self.locations.size
        self.nodeNo = N
        self.system = np.zeros((N,N))
        self.rhs = np.zeros(N)
        self.h = np.ones(N) * hIni
        self.conv = np.zeros(N)
        self.diff = np.zeros(N)
        self.Nind = np.arange(N - 2, -1, -1, dtype=int)
        self.soln = np.ones(N) * qIni
        self.mannings = np.ones(N) * mannings
        self.channelWidth = 10
        self.slope = np.ones(N) * slope
        self.diffLimit = 120
        # self.USBC = np.loadtxt(lines[25])
        # self.DSBC = np.loadtxt(lines[28])
        # self.source = np.loadtxt(lines[31])
        # self.boundaryType = lines[34]
        # self.timeScheme = lines[37]
        # self.printStep = lines[40]
        # self.outputFolder = lines[43]
        # self.rbfType = lines[46]
        # self.beta = lines[49]
        # self.isAugment = lines[52]


        for i in self.Nind:
            wettedPerimeter = self.channelWidth + 2 * np.sqrt(self.h[i] ** 2 + (self.h[i] / 2) ** 2)
            csArea = self.h[i] * self.channelWidth + self.h[i] ** 2 / 2
            R = csArea / wettedPerimeter
            S_f = self.mannings[i] ** 2 / (csArea * R ** (2 / 3)) ** 2 * self.soln[i] ** 2 ## constant mannings
            self.diff[i] = np.min([self.diffLimit, (np.abs(self.soln[i]) / 2 / (S_f) / self.channelWidth)])
            self.conv[i] = 5 * (S_f) ** .3 * np.abs(self.soln[i]) ** .4 / 3 / self.channelWidth ** .4 / self.mannings[i] ** .6

    def buildMQ(self, shapeParameter=-1):

        N = self.nodeNo
        r = np.zeros((N,N))
        phiHat = np.zeros((N,N))
        phiHat_x = np.zeros((N,N))
        phiHat_xx = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                r[i, j] = np.abs(self.locations[i] - self.locations[j])
                phiHat[i,j] = np.sqrt( r[i,j]**2 + shapeParameter**2 )
                phiHat_x[i, j] = (self.locations[i] - self.locations[j]) / phiHat[i, j]
                phiHat_xx[i, j] = 1 / phiHat[i, j] - (self.locations[i] - self.locations[j]) ** 2 / phiHat[i, j] ** 3
        self.f = phiHat
        self.fx = phiHat_x
        self.fxx = phiHat_xx
        self.invPnns = np.linalg.pinv(self.f)

    def buildTPS(self, beta=2):
        N = self.nodeNo
        phiHat = np.zeros((N,N))
        phiHat_x = np.zeros((N,N))
        phiHat_xx = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                if i != j:
                    r = np.sqrt((self.locations[i] - self.locations[j]) ** 2)
                    phiHat[i,j] = r**beta*np.log(r)
                    phiHat_x[i, j] = (self.locations[i]-self.locations[j])*r**(beta-2)*(beta*np.log(r)+1)
                    phiHat_xx[i, j] = r**(beta-2)*(beta*np.log(r)+1)+(self.locations[i]-self.locations[j])**2*r**(beta-4)*(2*(beta-1)+beta*(beta-2)*np.log(r))
        self.f = phiHat
        self.fx = phiHat_x
        self.fxx = phiHat_xx
        self.invPnns = np.linalg.pinv(self.f)

    def augment(self):
        """polynomial augmentation will come here"""
        pass

    def diffusion(self, D):
        """
        :param boundaries:
        :param D: diffusion coeffs, 1d array
        :return:
        """
        N = self.nodeNo
        for i in range(1, N-1):
            for j in range(N):
                self.system[i,j] -= D[i] * self.fxx[i,j]

    def advection(self, C):
        """
        :param boundaries:
        :param D: diffusion coeffs, 1d array
        :return:
        """
        N = self.nodeNo
        for i in range(1, N-1):
            for j in range(N):
                self.system[i,j] += C[i] * self.fx[i,j]

    def advectionDiffusion(self, C, D):
        self.advection(C)
        self.diffusion(D)

    def advectionDiffusionV2(self):
        N = self.nodeNo
        self.system[1:-1,:] = np.zeros((N-2,N))
        for i in range(1, N-1):
            for j in range(N):
                self.system[i,j] = -self.diff[i] * self.fxx[i,j] + self.conv[i] * self.fx[i, j]

    def solveUnsteady(self, source, initialValue, boundaryValue1Path, boundaryValue2Path, boundaryType=[0,1], timeScheme="backward", dt=0.01, printStep=100, printFolder="outputs"):
        """

        :param source:
        :param initialValue: 1d array, size of the array is N
        :param boundaryValue1,2: 1d arrays for boundary values at the ends, size is the same as the time iteration
        :param boundaryType:
        :param timeScheme:
        :param dt:
        :param endTime:
        :return:
        """
        self.printStep = printStep
        us = np.loadtxt(boundaryValue1Path)
        ds = np.loadtxt(boundaryValue2Path)
        timeIter = us.size

        if timeScheme=='backward':

            solnOld = initialValue
            self.soln = initialValue
            self.system = self.f + dt * self.system
            if boundaryType[0] == 1:
                self.system[0,:] = self.fx[0,:]
            else:
                self.system[0, :] = self.f[0, :]
            if boundaryType[1] == 1:
                self.system[-1,:] = self.fx[-1,:]
            else:
                self.system[-1, :] = self.f[-1, :]
            self.sysInv = np.linalg.pinv(self.system)

            for tt in range(1, timeIter):
                self.rhs[1:-1] = source[1:-1] * dt + solnOld[1:-1]
                self.rhs[0], self.rhs[-1] = us[tt], ds[tt]
                self.soln = np.matmul(self.f, np.matmul(self.sysInv, self.rhs))
                solnOld[:] =self.soln[:]

                time = tt * dt
                if printStep and tt%printStep==0:
                    np.savetxt(printFolder + "q" + str("{:.0f}".format(tt/printStep)) + ".txt", self.soln)

    def solveUnsteadyChannel(self, source, initialValue, boundaryValue1Path, boundaryValue2Path, boundaryType=[0,1], timeScheme="backward", dt=0.01, printStep=100, printFolder="outputs"):
        """

        :param source:
        :param initialValue: 1d array, size of the array is N
        :param boundaryValue1,2: 1d arrays for boundary values at the ends, size is the same as the time iteration
        :param boundaryType:
        :param timeScheme:
        :param dt:
        :param endTime:
        :return:
        """
        self.printStep = printStep
        us = np.loadtxt(boundaryValue1Path)
        ds = np.loadtxt(boundaryValue2Path)
        timeIter = us.size

        if timeScheme=='backward':

            solnOld = initialValue
            if boundaryType[0] == 1:
                self.system[0,:] = self.fx[0,:]
            else:
                self.system[0, :] = self.f[0, :]
            if boundaryType[1] == 1:
                self.system[-1,:] = self.fx[-1,:]
            else:
                self.system[-1, :] = self.f[-1, :]

            for tt in range(1, timeIter):
                self.advectionDiffusionV2()
                self.system[1:-1,:] = self.f[1:-1,:] + dt * self.system[1:-1,:]
                self.sysInv = np.linalg.pinv(self.system)
                self.rhs[1:-1] = source[1:-1] * dt + solnOld[1:-1]
                self.rhs[0], self.rhs[-1] = us[tt], ds[tt]
                self.soln = np.matmul(self.f, np.matmul(self.sysInv, self.rhs))
                solnOld[:] = self.soln[:]

                self.calculateH()

                time = tt * dt
                if printStep and tt%printStep==0:
                    np.savetxt(printFolder + "q" + str("{:.0f}".format(tt/printStep)) + ".txt", self.soln)
                    np.savetxt(printFolder + "h" + str("{:.0f}".format(tt / printStep)) + ".txt", self.h)

    def calculateH(self):
        """this is a trapezoidal channel"""
        wettedPerimeter = self.channelWidth + 2 * np.sqrt(self.h[-1] ** 2 + (self.h[-1] / 2) ** 2)
        csArea = self.h[-1] * self.channelWidth + self.h[-1] ** 2 / 2
        R = csArea / wettedPerimeter
        S_fForward = self.mannings[-1] ** 2 / (csArea * R ** (2 / 3)) ** 2 * self.soln[-1] ** 2
        for i in self.Nind:
            wettedPerimeter = self.channelWidth + 2 * np.sqrt(self.h[i] ** 2 + (self.h[i] / 2) ** 2)
            csArea = self.h[i] * self.channelWidth + self.h[i] ** 2 / 2
            R = csArea / wettedPerimeter
            # S_f = mannings[i] ** 2 / (csArea * R ** (2 / 3)) ** 2 * self.soln[i] ** 2
            S_f = self.mannings[i] ** 2 / (csArea * R ** (2 / 3)) ** 2 * self.soln[i] ** 2 ## constant mannings
            self.diff[i] = np.min([self.diffLimit, (np.abs(self.soln[i]) / 2 / (S_f) / self.channelWidth)])
            self.conv[i] = 5 * (S_f) ** .3 * np.abs(self.soln[i]) ** .4 / 3 / self.channelWidth ** .4 / self.mannings[i] ** .6
            self.h[i] = self.h[i + 1] + (self.slope[i] + (S_f + S_fForward)/2) * (self.locations[i+1] - self.locations[i])
            S_fForward = S_f
