import matplotlib.pyplot as plt
import numpy as np

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

    def __init__(self, dt, locs, cour, peclet, Qini, us, ds, rbfT, betty, simmy):

        self.dt = dt
        self.locations = locs
        N = self.locations.size
        self.nodeNo = N
        self.system = np.zeros((N,N))
        self.rhs = np.zeros(N)
        self.Nind = np.arange(N - 2, -1, -1, dtype=int)
        self.soln = np.ones(N) * Qini
        self.USBC = us
        self.DSBC = ds
        self.source = np.zeros(N)
        self.boundaryType = [0,1]
        self.timeScheme = "backward"
        self.printStep = 1
        self.rbfType = rbfT
        self.beta = betty
        self.simTime = simmy


        dx = np.abs(np.mean(self.locations[1:]-self.locations[:-1]))
        self.courant = cour
        self.peclet = peclet

        self.conv = np.ones(N) * cour * dx / dt
        self.diff = np.ones(N) * cour/peclet*dx/dt

        if self.rbfType == "MQ":
            self.buildMQ(self.beta)
        elif self.rbfType =="TPS":
            self.buildTPS(self.beta)


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


                time = tt * self.dt
                if tt % self.printStep == 0:
                    self.dsq = np.append(self.dsq, self.soln[-1])
                    self.usq = np.append(self.usq, self.soln[0])
                    self.time = np.append(self.time, time)

"""setting up the input variables"""
for dt in [1000, 500, 250, 100, 50]:
    for courant in [0.8]:
        for peclet in [1000000, 1]:
            x = np.linspace(0,20000, 21)

            endTime = 25 #hr
            timeIter = int(25 * 60 * 60 / dt)
            us = np.ones(timeIter) * 20
            ds = np.zeros(timeIter)

            for iter in range(timeIter):
                t = dt * iter
                if t >= 2 * 60 * 60 and t < 3 * 60 * 60:
                    us[iter] = 20 + 5 * (t - 2*60*60) / 60 / 60
                elif t  >= 3 * 60 * 60 and t  < 4 * 60 * 60:
                    us[iter] = 25 - 5 * (t - 3 * 60 * 60) / 60 / 60

            """This is where the solution starts"""
            soln = SingleChannel( dt, x, courant, peclet, 20, us, ds, "TPS", 4, 90000)
            soln.solveUnsteadyChannel()
            soln.time=soln.time/3600
            plt.plot(soln.time, soln.dsq, "--", label=f"dt: {dt:.0f}, Courant: {soln.courant:.02f}, Peclet: {soln.peclet:.04f}")

plt.plot(soln.time, soln.usq, color="k")
plt.legend()
plt.grid()
plt.show()