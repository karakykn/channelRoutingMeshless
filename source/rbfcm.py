import numpy as np
from scipy.linalg import block_diag

class Network(object):
    def __init__(self, inputGeneralPath):
        """channel count
        by that channel count open an iteration, read the files"""
        with open(inputGeneralPath, "r") as file:
            lines = file.readlines()
        self.dt = float(lines[1][:-1])
        self.diffLimit = float(lines[4][:-1])
        self.timeScheme = lines[7][:-1]
        self.printStep = float(lines[10][:-1])
        self.outputFolder = lines[13][:-1]
        self.rbfType = lines[16][:-1]
        self.beta = float(lines[19][:-1])
        self.isAugment = float(lines[22][:-1])
        self.noChannels = int(lines[25][:-1])
        self.connections = []
        self.conLocs = []
        self.conLocsInd = []
        self.channel = []

        for i in range(self.noChannels):
            self.channel.append(SingleChannel(lines[28 + 3 * i + 2 * (self.noChannels)][:-1], inputGeneralPath))

        for i in range(self.noChannels - 1):
            self.connections.append(eval(lines[28 + 2 * i][:-1]))
            # self.conLocs.append(float(lines[28 + (2*i + 1)][:-1]))
            # self.conLocsInd.append(np.where(self.channel[self.connections[i][0]].locations == self.conLocs[i])[0][0])

        file.close()

    def connectChannels(self, timeIter):
        for i in self.channel:
            i.advectionDiffusion(self.dt)
            i.updateBC()
            i.buildRHS(timeIter)

        sne = np.zeros((self.noChannels, 2), dtype = int)
        sne[0, 1] = self.channel[0].nodeNo
        for i in range(1, self.noChannels):
            sne[i, 0] = sne[i-1, 1]
            sne[i, 1] = sne[i, 0] + self.channel[i].nodeNo
        self.globalMatrix = np.zeros((sne[-1, 1], sne[-1, 1]))
        self.globalRHS = np.zeros(sne[-1, 1])
        for i in range(self.noChannels):
            self.globalMatrix[sne[i, 0]:sne[i, 1], sne[i, 0]:sne[i, 1]] = self.channel[i].system
            self.globalRHS[sne[i, 0]:sne[i, 1]] = self.channel[i].rhs
        for con in self.connections:
            self.globalMatrix[sne[con[0],0], sne[con[0],0]: sne[con[0],1]] = self.channel[con[0]].f[sne[con[0],0], :]
            self.globalMatrix[sne[con[0],0], sne[con[1], 0]: sne[con[1], 1]] = -self.channel[con[1]].f[-1, :]
            self.globalRHS[sne[con[0],0]] = 0
        self.sne = sne

    def redistributeQ(self):
        for i in range(self.noChannels):
            self.channel[i].Q = self.globalSoln[self.sne[i,0]:self.sne[i,1]]


    def solveUnstedyNetwork(self):

        timeIter = self.channel[0].DSBC.size
        if self.timeScheme == "backward":

            for tt in range(1, timeIter):
                self.connectChannels(tt)
                self.invGlobal = np.linalg.pinv(self.globalMatrix)
                self.globalSoln = np.matmul(self.invGlobal, self.globalRHS)

                self.redistributeQ()

                for i in range(self.noChannels):
                    self.channel[i].calculateH()
                    for c in self.connections:
                        if i == c[0]:
                            self.channel[c[1]].h[:] = self.channel[c[0]].h[0]

                for i in range(self.noChannels):
                    np.savetxt(self.outputFolder + 'channel' + str(i) + 'Q' + str(tt) +'.txt', self.channel[i].Q)
                    np.savetxt(self.outputFolder + 'channel' + str(i) + 'h' + str(tt) +'.txt', self.channel[i].h)








    """add calculate h function here, it will be similar to the one in single channel class."""

def buildMQ(channel):
    shapeParameter = channel.beta
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
            phiHat_xx[i, j] = 1 / phiHat[i, j] - (channel.locations[i] - channel.locations[j]) ** 2 / phiHat[i, j] ** 3
    channel.f = phiHat
    channel.fx = phiHat_x
    channel.fxx = phiHat_xx

def buildTPS(channel):
    beta = channel.beta
    N = channel.nodeNo
    phiHat = np.zeros((N, N))
    phiHat_x = np.zeros((N, N))
    phiHat_xx = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                r = np.sqrt((channel.locations[i] - channel.locations[j]) ** 2)
                phiHat[i, j] = r ** beta * np.log(r)
                phiHat_x[i, j] = (channel.locations[i] - channel.locations[j]) * r ** (beta - 2) * (beta * np.log(r) + 1)
                phiHat_xx[i, j] = r ** (beta - 2) * (beta * np.log(r) + 1) + (
                            channel.locations[i] - channel.locations[j]) ** 2 * r ** (beta - 4) * (
                                              2 * (beta - 1) + beta * (beta - 2) * np.log(r))
    channel.f = phiHat
    channel.fx = phiHat_x
    channel.fxx = phiHat_xx
    # channel.invPnns = np.linalg.pinv(channel.f)

def augment(channel):
    """polynomial augmentation will come here"""
    pass

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

    def __init__(self, inputFilePath, inputGeneralPath):

        with open(inputGeneralPath, "r") as file:
            lines =file.readlines()
        self.dt = float(lines[1][:-1])
        self.diffLimit = float(lines[4][:-1])
        self.timeScheme = lines[7][:-1]
        self.printStep = float(lines[10][:-1])
        self.outputFolder = lines[13][:-1]
        self.rbfType = lines[16][:-1]
        self.beta = float(lines[19][:-1])
        self.isAugment = float(lines[22][:-1])

        file.close()

        with open(inputFilePath, "r") as file:
            lines = file.readlines()

        self.locations = np.loadtxt(lines[1][:-1])
        N = self.locations.size
        self.nodeNo = N
        self.system = np.zeros((N,N))
        self.rhs = np.zeros(N)
        self.h = np.loadtxt(lines[16][:-1])
        self.conv = np.zeros(N)
        self.diff = np.zeros(N)
        self.Nind = np.arange(N - 2, -1, -1, dtype=int)
        self.Q = np.loadtxt(lines[13][:-1])
        self.mannings = np.loadtxt(lines[4][:-1])
        self.channelWidth = np.loadtxt(lines[10][:-1])
        self.slope = np.loadtxt(lines[7][:-1])
        self.USBC = np.loadtxt(lines[19][:-1])
        self.DSBC = np.loadtxt(lines[22][:-1])
        self.source = np.loadtxt(lines[25][:-1])
        self.boundaryType = eval(lines[28][:-1])
        self.channelSideSlope = float(lines[31][:-1])

        file.close()

        for i in self.Nind:
            wettedPerimeter = self.channelWidth[i] + 2 * np.sqrt(self.h[i] ** 2 + (self.h[i] / 2) ** 2)
            csArea = self.h[i] * self.channelWidth[i] + self.h[i] ** 2 / 2
            R = csArea / wettedPerimeter
            S_f = self.mannings[i] ** 2 / (csArea * R ** (2 / 3)) ** 2 * self.Q[i] ** 2 ## constant mannings
            self.diff[i] = np.min([self.diffLimit, (np.abs(self.Q[i]) / 2 / (S_f) / self.channelWidth[i])])
            self.conv[i] = 5 * (S_f) ** .3 * np.abs(self.Q[i]) ** .4 / 3 / self.channelWidth[i] ** .4 / self.mannings[i] ** .6

        if self.rbfType == "MQ":
            buildMQ(self)
        elif self.rbfType =="TPS":
            buildTPS(self)

        if self.isAugment == 1:
            self.augment()

    def advectionDiffusion(self, dt):
        N = self.nodeNo
        for i in range(N):
            for j in range(N):
                self.system[i,j] = -self.diff[i] * self.fxx[i,j] + self.conv[i] * self.fx[i, j]
        self.system = self.f + dt * self.system

    def updateBC(self):
        if self.boundaryType[0] == 0:
            self.system[0,:] = self.f[0,:]
        elif self.boundaryType[0] == 1:
            self.system[0, :] = self.fx[0,:]
        else:
            pass

        if self.boundaryType[1] == 0:
            self.system[-1,:] = self.f[-1,:]
        elif self.boundaryType[1] == 1:
            self.system[-1, :] = self.fx[-1,:]
        else:
            pass

    def buildRHS(self, timeIter):
        self.rhs = self.source * self.dt + self.Q
        if self.boundaryType[0] != 2:
            self.rhs[0] = self.USBC[timeIter]
        if self.boundaryType[1] != 2:
            self.rhs[-1] = self.DSBC[timeIter]


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
        us = self.USBC
        ds = self.DSBC
        timeIter = us.size
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

            for tt in range(1, timeIter + 1):
                self.advectionDiffusion()
                self.system[1:-1,:] = self.f[1:-1,:] + self.dt * self.system[1:-1,:]
                self.sysInv = np.linalg.pinv(self.system)
                self.rhs[1:-1] = self.source[1:-1] * self.dt + solnOld[1:-1]
                self.rhs[0], self.rhs[-1] = us[tt], ds[tt]
                self.soln = np.matmul(self.f, np.matmul(self.sysInv, self.rhs))
                solnOld[:] = self.soln[:]

                self.calculateH()

                time = tt * self.dt
                if tt % self.printStep == 0:
                    np.savetxt(self.outputFolder + "q" + str("{:.0f}".format(tt/self.printStep)) + ".txt", self.soln)
                    np.savetxt(self.outputFolder + "h" + str("{:.0f}".format(tt / self.printStep)) + ".txt", self.h) \

    def calculateH(self):
        """channelSideSlope=0 means rectangular channel, this part can be directly supplied with the actual values of the channel"""
        wettedPerimeter = self.channelWidth[-1] + 2 * np.sqrt(self.h[-1] ** 2 + (self.h[-1] * self.channelSideSlope) ** 2)
        csArea = self.h[-1] * self.channelWidth[-1] + self.h[-1] * (self.h[-1] * self.channelSideSlope)
        R = csArea / wettedPerimeter
        S_fForward = self.mannings[-1] ** 2 / (csArea * R ** (2 / 3)) ** 2 * self.Q[-1] ** 2
        for i in self.Nind:
            wettedPerimeter = self.channelWidth[i] + 2 * np.sqrt(self.h[i] ** 2 + (self.h[i] * self.channelSideSlope) ** 2)
            csArea = self.h[i] * self.channelWidth[i] + self.h[i] * (self.h[i] * self.channelSideSlope)
            R = csArea / wettedPerimeter
            S_f = self.mannings[i] ** 2 / (csArea * R ** (2 / 3)) ** 2 * self.Q[i] ** 2
            self.diff[i] = np.min([self.diffLimit, (np.abs(self.Q[i]) / 2 / (S_f) / self.channelWidth[i])])
            self.conv[i] = 5 * (S_f) ** .3 * np.abs(self.Q[i]) ** .4 / 3 / self.channelWidth[i] ** .4 / self.mannings[i] ** .6
            self.h[i] = self.h[i + 1] + (self.slope[i] + (S_f + S_fForward)/2) * (self.locations[i+1] - self.locations[i])
            S_fForward = S_f