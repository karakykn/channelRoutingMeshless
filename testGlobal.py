import numpy as np
from scipy.linalg import block_diag


class Channel:
    def __init__(self, system_matrix):
        self.system = system_matrix
        self.nodeNo = self.system.shape[0]
        self.Q = np.zeros(self.system.shape[0])


class Network:
    def __init__(self, channels, connections, conLocsInd):
        self.channel = channels
        self.connections = connections
        self.conLocsInd = conLocsInd
        self.noChannels = len(self.channel)
        self.RHS = np.array([1, 1, 1, 1, 1, 1, 1, 1])

        self.connectChannels()
        self.solve()
        self.redistributeQ()

    def connectChannels(self):
        sne = np.zeros((self.noChannels, 2))
        sne[0, 1] = self.channel[0].nodeNo
        for i in range(1, self.noChannels):
            sne[i, 0] = sne[i-1, 1]
            sne[i, 1] = sne[i, 0] + self.channel[i].nodeNo
        self.globalMatrix = np.zeros((sne[-1, 1], sne[-1, 1]))
        for i in range(self.noChannels):
            self.globalMatrix[sne[i, 0]:sne[i, 1]] = self.channel[i].system
        for con in self.connections:
            self.globalMatrix[sne[con[0]], sne[con[0],0]: sne[con[0],1]] = self.channel[con[0]].f[sne[con[0]], :]
            self.globalMatrix[sne[con[0]], sne[con[1], 0]: sne[con[1], 1]] = -self.channel[con[1]].f[sne[con[1]], :]
        self.sne = sne



    def solve(self):
        self.invGlob = np.linalg.pinv(self.globalMatrix)
        self.solns = np.matmul(self.invGlob, self.RHS)

    def redistributeQ(self):
        for i in self.channel:
            i.Q = self.solns[self.sne[i, 0]: self.sne[i, 1]]


# Example case
ch1_matrix = np.array([[1, 0.1, 0.1],
                       [0.1, 1, 0.1],
                       [0.1, 0.1, 1]])

ch2_matrix = np.array([[2, 0.2, .1],
                       [0.2, 2, .2],
                       [0.15, .2, 2]])

ch3_matrix = np.array([[0.2, .1],
                       [0.2, .2]])


channels = [Channel(ch1_matrix), Channel(ch2_matrix), Channel(ch3_matrix)]
# channels = [Channel(ch1_matrix, [2, 1]), Channel(ch2_matrix, [0, 2])]
connections = [[0, 1], [0, 2]]  # Channel 1 connects to Channel 0
conLocsInd = [0, 0] # Channel 1 connects at the first point of Channel 0

network = Network(channels, connections, conLocsInd)
print(network.solns)
print(network.channel[0].Q)
print(network.channel[1].Q)
print(network.channel[2].Q)
