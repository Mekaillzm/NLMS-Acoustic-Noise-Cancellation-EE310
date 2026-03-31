import numpy as np

class AlgoNLMS:
    def __init__(self):
        '''Initialises the algorithm.

        Sets:
        -filter length (N) as N
        -step size (μ) as stepSize
        -regularization parameter (δ) as regParam
        -threshold for filter, c
        -stop timer in number of samples as halt
        -filter ( w(n) ) as w
        -buffer array ( x(n) ) as x
        

        '''
        self.N = 1023
        '''
        
        N is the filter length. This is an integer that is recommended to vary between 512 - 2048 taps.

        It sets the size of the filter and buffer arrays, deciding how long the sound signal that is processed should be 

        Allowed values: 511, 1023, 2047
        Odd number chosen to ensure the filter has an exact centre point.
        Smaller value: lower latency, lower resolution
        Larger value: greater latency, greater resolution

        1023 is in-between, so this was chosen to balance latency and resolution.
        '''
        
        self.stepSize = 0.4
        '''
        Step size is the rate of learning of the algorithm.
        It sets how aggressively the filter updates.

        Recommended values: 0.3 - 0.7
        '''

        self.regParam = 0.00001
        '''
        Regularisation parameter: small value that is added to the denominator of the NLMS equation to prevent division by zero and maintain stability.

        Smaller regularisation parameter: faster convergence/learning, lower stability
        Larger regularisation parameter: slower convergence/learning, higher stability
        
        Recommended values: 10^−6 - 10^−4

        10^-5 chosen as a mid range value
        '''

        self.c = 0.65
        '''
        This is the threshold for checkState.
        Recommended values: 0.5 - 0.7
        '''

        self.halt = 1600
        self.curHalt = 0
        '''
        This is how many samples to wait until continuing.
        halt represents the baseline amount
        curHalt represents how many samples to currently wait for

        It is used in checkState in case of double talk
        '''

        self.w = np.zeros(self.N)
        self.x = np.zeros(self.N)

        '''w(n) and x(n) are N x 1 arrays initialised to zero values, representing the filter and buffer'''
    
        self.fs = 1600 #placeholder for sample rate in Hz
        self.nStep = 0 #time tracking variable for the generated signal
    def genSample(self):
        '''
        Generates a sample signal in real time for each time step.
        This sample signal is a set of sine waves and is for basic testing only.

        The far end sample is treated as a 400Hz sine wave.

        The near end signal is an amplitude-scaled signal at 1kHz

        Noise is added with np.random.normal as normally distributed random numbers.
        '''

        t = self.nStep / self.fs #The step per rate, representing the current time
        xn = np.sin(2 * np.pi * 400 * t) #setting the far end sample


        #x(n) = sin(400 * 2pi t)

        # using the equation d(n) = echo + near_end + noise
        echon = 0.5 * xn #echo as x(n), with half the amplitude
        if t>=0.5:
            nearEndn = 0.3 * np.sin(np.pi * 1000 * t) #near end signal
        else: nearEndn = 0 #initially set the near end signal to 0, so that the model can train

        #near end signal as 0.3sin(2pi * 1000t)

        noisen = np.random.normal(0, 0.05) # noise signal
        '''
        noise signal as a random number from a normal distribution
        #mean = 0, std = 0.05
        By using a zero-mean normally distributed random variable, noise can be introduced, but it remains stable and 
        close to zero. 
        '''

        dn = echon + nearEndn + noisen
        # d(n) = echo(n) + near_end(n) + noise(n)

        self.nStep+=1 #increment the simulated time

        return xn, dn #return a tuple with x(n) and d(n)
    
    def updateBuffer(self, xn):
            '''This function shifts all the values in x(n) by 1 to the right.
            The last value is deleted, and a new value is inserted at position 0.
            The new value is the sample. E.g. it can be obtained from genSample()'''
            self.x[1:] = self.x[:-1] 
            #x[:-1] slices off the last position
            #x[1:] pushes the values by 1 to the right

            self.x[0] = xn #index at 0 is set to the new sample 
            
    def estEcho(self):
         '''Calculates the dot product of w(n) and x(n) as an estimation of the echo.'''

         return np.dot(self.w, self.x) #dot product
    
    def calcError(self, dn, yEst):
         
              '''Subtracts the estimated echo (yEst) from d(n), the microphone signal
              This is to obtain e(n), the error'''
              self.en = dn - yEst # en = d(n) - yEst

    def checkState(self, dn):
          '''
          This function implements a Geigel Double Talk Detector

          It checks for the follwing practical conditions:
          -Far end only - returns true
          -near end only - returns false
          -double talk - returns false

          Far end only is the ideal condition, when only the far end signal is coming in
          Near end only is less ideal, for this, the filter adaptation freezes

          This is because having no far end signal means the signal has no ideal signal to learn from

          In the case of double talk, learning is also stopped as the signal would appear as an unpredictable mess.


          Implementation: the code gets the highest magnitude value in the far-end signal, x.
          This value is scaled by c, the threshold (defined in __init__).

          If the mic signal has an amplitude greater than the max value, that means there is double talk.
          It also detects the near end only case, as the highest magnitude value of x is automatically 0.
          Thus, any non-zero signal in d(n) will always be greater than x(n), which is zero

          If any of these conditions are active, a delay is introduced
          If the no delay is present and all conditions are inactive (far end only case), then the algorithm is allowed to proceed

          '''

          maxX = np.max(np.abs(self.x))
          #this is the value in x(n) with the greatest magnitude

          if np.abs(dn) >= self.c * maxX:#check if the scaled greatest magnitude value in x is less than the magnitude of the mic signal
                self.curHalt = self.halt

                #reset the wait time, curHalt, to its base value to pause the algorithm
          if self.curHalt > 0: #check if halted
                self.curHalt -= 1
                return False #stop the algortihm from updating for this sample
          
          return True #if all the conditions pass then the algorithm is not halted
    
    
    def updateWeights(self):
          '''This updates all the filter weights by using the NLMS algorithm
          
          It does this by calculating the energy of the buffer by squaring x to get ||x||^2
          
          The regularisation parameter is added to the energy for stability and to prevent division by zero
          
          then, the step size + error(n) are divided by the previous energy sum
          
          This gives an update factor that is used to recalibrate the weight, w
          
          Equation: w(n+1) = w(n) + (mu / (||x(n)||^2 + delta)) * e(n) * x(n)
          '''

          energy = np.dot(self.x, self.x) + self.regParam #energy + regularisation parameter
          # ||x||^2 + regularisation parameter
          update = (self.stepSize * self.en)/energy #(mu / (||x(n)||^2 + delta)) * e(n)

          self.w = self.w + update * self.x #update x(n+1)
          #w(n+1) = w(n) + (mu / (||x(n)||^2 + delta)) * e(n) * x(n)
