import numpy as np
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as pyp
from scipy.integrate import trapz

class Algae_bacteria_model():
    def __init__(self):
        # Biokinetics
        self.ubac = .9197#1.374#2.195#0.42
        self.ualg = 1.083#1.105#1.512#1.07
        self.Knbac = 4.176#4.3#1.055#1
        self.Knalg = 35.29#48.58#2.817#0.1
        self.KO2 = .01958#2.548#.4687#0.4
        self.Kco2 = .9565#10.11#.4318#5.6*10**-3
        self.bbac = .4057#1e-6#.005769#0.05
        self.balg = 1e-6#1e-6#1e-6#0.1
        self.KLao2 = 4
        self.S02sat = 8.58
        self.I = 100
        self.kI = .06454#11.17#1e-6#0.1

        #Stoichiometry
        self.falgc = 0.383
        self.falgn = 0.065
        self.fbacC = 1.375
        self.ixbac = 0.08
        self.ybac = 1.018#.9682#.8471#0.25
        self.yalgnh4c = 0.842
        self.yalgnh4n = 11.91
        self.yalgnh4o = 0.996
        self.yalgno3c = 0.622
        self.yalgno3n = 3.415
        self.yalgno3o = 1.301

    def ode(self,t,u):
        
        co2,nh4,no3,o2,alg,bac = u
        #DIFFERENTIAL Equations for process rates
        dAnh = self.ualg * (self.I / (self.kI + self.I)) * co2 / (self.Kco2 + co2) * nh4 / (self.Knalg + nh4) * alg

        dAno = self.ualg * (self.I / (self.kI + self.I)) * co2 / (self.Kco2 + co2) * no3 / (self.Knalg + no3) * self.Knalg / (self.Knalg + nh4) * alg
        
        dAdecay = self.balg * alg

        dB = self.ubac*nh4 / (self.Knbac + nh4) * o2 / (self.KO2 + o2) * bac

        dBdecay = self.bbac * bac

        do2atm = self.Knalg * (self.S02sat - o2)

        #Calculates the reaction rates of the process

        dco2 = np.sum([-1/self.yalgnh4c * dAnh,-1/self.yalgno3c * dAno, self.falgc * dAdecay, self.fbacC * dB],axis = 0)

        dnh4 = np.sum([-1/self.yalgnh4n * dAnh,self.falgn*dAdecay,(-1*self.ixbac-self.ybac) * dB, self.ixbac * dBdecay],axis = 0)

        dno3 = np.sum([-1/self.yalgno3n * dAno,1/self.ybac * dB],axis = 0)

        do2 = np.sum([self.yalgnh4o * dAnh,self.yalgno3o * dAno, -1*(4.57 - self.ybac)/self.ybac * dB,do2atm],axis = 0)
        
        dalg = np.sum([dAnh,dAno,-1*dAdecay],axis = 0)

        dbac = np.sum([dB,-1*dBdecay],axis = 0)

        return [dco2,dnh4,dno3,do2,dalg,dbac]
    
    def main(self):

        #intial guesses
        #co2,nh4,no3,o2,alg,bac
        vars = np.array([0.1,26.9,2,7.2,112.2,5])/1000

        t_span = (0,6)
        t_eval = np.linspace(t_span[0],t_span[1],101)

        solution = solve_ivp(self.ode,t_span,vars,method = 'LSODA')

        print(solution)

        names = ['CO2','NH4','NO3','O2','Algae','Bac']

        for i in range(len(names)):
            pyp.subplot(2,3,i+1)
            pyp.ylabel(names[i])
            pyp.xlabel('time')
            pyp.plot(solution.t,solution.y[i,:]*1000)
            print(solution.y[i,:])
            
        pyp.show()

    def residuals(self,params):

        self.ybac,self.ubac,self.ualg,self.Knbac,self.Knalg,self.KO2,self.Kco2,self.bbac,self.balg,self.KLao2,self.kI = params

        vars = np.array([0.1,26.9,2,7.2,112.2,5])

        t_span = (0,600)
        t_eval = np.linspace(t_span[0],t_span[1],)

        solution = solve_ivp(self.ode,t_span,vars,method = 'LSODA',t_eval= [0,1,2,3,4,5,6])
        solution.y = solution.y

        return solution
    
#ab = Algae_bacteria_model()
#ab.main()