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

    def __init__(self, dt, locs, C, D, Qini, us, ds, simmy, pltY):

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
        self.conv = np.ones(N) * C
        self.diff = np.ones(N) * D
        self.source = np.zeros(N)
        self.boundaryType = [0,1]
        self.timeScheme = "backward"
        self.printStep = 1
        self.simTime = simmy
        self.pltY = pltY


        dx = np.abs(np.mean(self.locations[1:]-self.locations[:-1]))
        self.courant = C * dt / dx
        self.peclet = C / D
        print(f'Courant number: {self.courant:0.2f}')
        print(f'Peclet number: {self.peclet:0.4f}')
        print(f'Celerity number: {np.mean(self.conv):0.2f}')
        print(f'Diffusivity number: {np.mean(self.diff):0.0f}')


    def advectionDiffusionV2(self):
        N = self.nodeNo
        self.system[1:-1,:] = np.zeros((N-2,N))
        for i in range(1, N-1):
            self.system[i, i] = 1 + 2 * self.diff[i] * self.dt / (self.locations[i] - self.locations[i-1])**2 + self.conv[i] * self.dt / (self.locations[i] - self.locations[i-1])
            self.system[i,i-1] = -self.diff[i] * self.dt / (self.locations[i] - self.locations[i-1])**2 - self.conv[i] * self.dt / (self.locations[i] - self.locations[i-1])
            self.system[i,i+1] = -self.diff[i] * self.dt / (self.locations[i+1] - self.locations[i])**2
        self.system[0, 0] = 1
        self.system[-1, -1], self.system[-1, -2] = 1, -1

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
            plt.ion()
            for tt in range(1, timeIter):
                self.advectionDiffusionV2()
                self.sysInv = np.linalg.pinv(self.system)
                self.rhs[1:-1] = solnOld[1:-1]
                self.rhs[0], self.rhs[-1] = us[tt], ds[tt]
                self.soln = np.matmul(self.sysInv, self.rhs)
                solnOld[:] = self.soln[:]


                time = tt * self.dt
                if tt % self.printStep == 0:
                    self.dsq = np.append(self.dsq, self.soln[-1])
                    self.usq = np.append(self.usq, self.soln[0])
                    self.time = np.append(self.time, time)
                    if self.pltY == 1:
                        plt.plot(self.locations, self.soln)
                        plt.ylim(15, 25)
                        plt.grid()
                        plt.pause(.2)
                        plt.show()
                        plt.cla()
            plt.ioff()

"""setting up the input variables"""
for dt in [250]:
    for N in [81]:
        for cele in [.8]:
            for diffu in [150]:
                x = np.linspace(0,20000, N)
                meanx = np.mean(x)
                c = 4 * meanx

                endTime = 25 #hr
                timeIter = int(25 * 60 * 60 / dt)
                us = np.ones(timeIter) * 20
                ds = np.ones(timeIter) * 0

                for iter in range(timeIter):
                    t = dt * iter
                    if t >= 2 * 60 * 60 and t < 3 * 60 * 60:
                        us[iter] = 20 + 5 * (t - 2*60*60) / 60 / 60
                    elif t  >= 3 * 60 * 60 and t  < 4 * 60 * 60:
                        us[iter] = 25 - 5 * (t - 3 * 60 * 60) / 60 / 60

                """This is where the solution starts"""
                soln = SingleChannel(dt, x, cele, diffu, 20, us, ds, 90000, 0)
                soln.solveUnsteadyChannel()
                soln.time=soln.time/3600
                plt.plot(soln.time, soln.dsq, "--", label=f"dt: {dt:.0f}, Courant: {soln.courant:.02f}, Peclet: {soln.peclet:.04f}, C: {cele:.0f}, D: {diffu:.0f}")

plt.plot(soln.time, soln.usq, color="k")
plt.legend()
plt.grid()
plt.show()