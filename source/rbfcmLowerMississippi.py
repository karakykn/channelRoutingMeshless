import numpy as np
import os
from scipy.interpolate import interp1d

class SingleChannel(object):
    """Radial Basis function collocation method for 1D diffusive wave equation. Written by Ismet Karakan.
    For further questions: karakan@sc.edu

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

    def __init__(self, inputFilePath):

        with open(inputFilePath, "r") as file:
            lines = file.readlines()

        self.dt = float(lines[1][:-1])
        self.locations = np.loadtxt(lines[7][:-1])
        N = self.locations.size
        self.nodeNo = N
        self.system = np.zeros((N,N))
        self.rhs = np.zeros(N)
        self.h = np.loadtxt(lines[22][:-1])
        self.conv = np.zeros(N)
        self.diff = np.zeros(N)
        self.Nind = np.arange(N - 2, -1, -1, dtype=int)
        self.soln = np.loadtxt(lines[19][:-1])
        self.mannings = np.loadtxt(lines[10][:-1])
        self.channelWidth = np.loadtxt(lines[16][:-1])
        self.slope = np.loadtxt(lines[13][:-1])
        self.diffLimit = float(lines[4][:-1])
        self.USBC = np.loadtxt(lines[25][:-1])
        self.DSBC = np.loadtxt(lines[28][:-1])
        self.upstreamH = np.loadtxt(lines[61][:-1])
        self.source = np.loadtxt(lines[31][:-1])
        self.boundaryType = eval(lines[34][:-1])
        self.timeScheme = lines[37][:-1]
        self.printStep = float(lines[40][:-1])
        self.outputFolder = lines[43][:-1]
        self.rbfType = lines[46][:-1]
        self.beta = float(lines[49][:-1])
        self.isAugment = float(lines[52][:-1])
        self.channelSideSlope = float(lines[55][:-1])
        self.simTime = float(lines[58][:-1])
        self.riverP = np.loadtxt(lines[64][:-1], skiprows=1, delimiter=',')

        self.riverpH = self.riverP[:, 0]
        self.riverpArea = self.riverP[:, 2]
        self.riverpR = self.riverP[:, 3]

        self.Area_interpolator = interp1d(self.riverpH, self.riverpArea, kind='linear', fill_value="extrapolate")
        self.R_interpolator = interp1d(self.riverpH, self.riverpR, kind='linear', fill_value="extrapolate")

        file.close()

        for i in self.Nind:
            wettedPerimeter = self.channelWidth[i] + 2 * np.sqrt(self.h[i] ** 2 + (self.h[i] / 2) ** 2)
            csArea = self.h[i] * self.channelWidth[i] + self.h[i] ** 2 / 2
            R = csArea / wettedPerimeter
            S_f = self.mannings[i] ** 2 / (csArea * R ** (2 / 3)) ** 2 * self.soln[i] ** 2 ## constant mannings
            self.diff[i] = np.min([self.diffLimit, (np.abs(self.soln[i]) / 2 / (S_f) / self.channelWidth[i])])
            self.conv[i] = 5 * (S_f) ** .3 * np.abs(self.soln[i]) ** .4 / 3 / self.channelWidth[i] ** .4 / self.mannings[i] ** .6

        if self.rbfType == "MQ":
            self.buildMQ(self.beta)
        elif self.rbfType =="TPS":
            self.buildTPS(self.beta)

        if self.isAugment == 1:
            self.augment()

    def buildMQ(channel, shapeParameter=-1):
        N = channel.nodeNo
        r = np.zeros((N, N))
        phiHat = np.zeros((N, N))
        phiHat_x = np.zeros((N, N))
        phiHat_xx = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                r[i, j] = np.abs(channel.locations[i] - channel.locations[j])
                phiHat[i, j] = np.sqrt(r[i, j] ** 2 + shapeParameter ** 2)
                phiHat_x[i, j] = (channel.locations[i] - channel.locations[j]) / phiHat[i, j]
                phiHat_xx[i, j] = 1 / phiHat[i, j] - (channel.locations[i] - channel.locations[j]) ** 2 / phiHat[
                    i, j] ** 3
        channel.f = phiHat
        channel.fx = phiHat_x
        channel.fxx = phiHat_xx
        channel.invPnns = np.linalg.pinv(channel.f)

    def buildTPS(channel, beta=2):
        N = channel.nodeNo
        phiHat = np.zeros((N, N))
        phiHat_x = np.zeros((N, N))
        phiHat_xx = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if i != j:
                    r = np.sqrt((channel.locations[i] - channel.locations[j]) ** 2)
                    phiHat[i, j] = r ** beta * np.log(r)
                    phiHat_x[i, j] = (channel.locations[i] - channel.locations[j]) * r ** (beta - 2) * (
                                beta * np.log(r) + 1)
                    phiHat_xx[i, j] = r ** (beta - 2) * (beta * np.log(r) + 1) + (
                            channel.locations[i] - channel.locations[j]) ** 2 * r ** (beta - 4) * (
                                              2 * (beta - 1) + beta * (beta - 2) * np.log(r))
        channel.f = phiHat
        channel.fx = phiHat_x
        channel.fxx = phiHat_xx
        channel.invPnns = np.linalg.pinv(channel.f)

    def augment(channel):
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

    def solveUnsteadyChannel(self):
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
        self.dsq = np.array([self.soln[-1]])
        self.usq = np.array([self.soln[0]])
        self.time = np.array([0])
        us = self.USBC
        ds = self.DSBC
        timeIter = int(self.simTime / (self.dt))
        solnOld = np.zeros(self.nodeNo)

        if self.timeScheme=='backward':

            solnOld[:] = self.soln[:]
            if self.boundaryType[0] == 1:
                self.system[0,:] = self.fx[0,:]
            else:
                self.system[0, :] = self.f[0, :]
            if self.boundaryType[1] == 1:
                self.system[-1,:] = self.fx[-1,:]
            else:
                self.system[-1, :] = self.f[-1, :]

            for tt in range(1, timeIter):
                self.advectionDiffusionV2()
                self.system[1:-1,:] = self.f[1:-1,:] + self.dt * self.system[1:-1,:]
                self.sysInv = np.linalg.pinv(self.system)
                self.rhs[1:-1] = self.source[1:-1] * self.dt + solnOld[1:-1]
                self.rhs[0], self.rhs[-1] = us[tt], ds[tt]
                self.soln = np.matmul(self.f, np.matmul(self.sysInv, self.rhs))
                solnOld[:] = self.soln[:]

                self.h[0] = self.upstreamH[tt]

                self.calculateH()

                time = tt * self.dt
                if tt % self.printStep == 0:
                    print(f"Time: {self.time[-1]:.0f}s")
                    np.savetxt(self.outputFolder + "q" + f"{time:.0f}s" + ".txt", self.soln)
                    np.savetxt(self.outputFolder + "h" + f"{time:.0f}s" + ".txt", self.h)

                    self.dsq = np.append(self.dsq, self.soln[-1])
                    self.usq = np.append(self.usq, self.soln[0])
                    self.time = np.append(self.time, time)

            np.savetxt(self.outputFolder + "q" + f"{time:.0f}s" + ".txt", self.soln)
            np.savetxt(self.outputFolder + "h" + f"{time:.0f}s" + ".txt", self.h)

            np.savetxt(
                self.outputFolder + "downstreamQ" + "{:.0f}".format(self.locations[-1] / (self.nodeNo - 1)) + ".txt",
                self.dsq
            )
            np.savetxt(
                self.outputFolder + "upstreamQ" + "{:.0f}".format(self.locations[-1] / (self.nodeNo - 1)) + ".txt",
                self.usq
            )
            np.savetxt(
                self.outputFolder + "time" + ".txt",
                self.time
            )

    def calculateH(self):
        csArea = self.Area_interpolator(self.h[0])
        R = self.R_interpolator(self.h[0])
        S_fBackward = self.mannings[0] ** 2 / (csArea * R ** (2 / 3)) ** 2 * self.soln[0] ** 2
        for i in range(1,self.nodeNo):
            csArea = self.Area_interpolator(self.h[i])
            R = self.R_interpolator(self.h[i])
            S_f = self.mannings[i] ** 2 / (csArea * R ** (2 / 3)) ** 2 * self.soln[i] ** 2
            self.diff[i] = np.min([self.diffLimit, (np.abs(self.soln[i]) / 2 / (S_f) / self.channelWidth[i])])
            self.conv[i] = 5 * (S_f) ** .3 * np.abs(self.soln[i]) ** .4 / 3 / self.channelWidth[i] ** .4 / self.mannings[i] ** .6
            self.h[i] = self.h[i - 1] - (self.slope[i] + (S_f + S_fBackward)/2) * (self.locations[i] - self.locations[i-1])
            S_fBackward = S_f