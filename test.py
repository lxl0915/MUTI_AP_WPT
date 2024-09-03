import math
import numpy as np
"""
A_d = 4.11
d_e = 2.8
f_c = 915*10**6
#down = 
distance = A_d*math.pow((3*10**8)/(4*3.1415926*f_c*2), d_e)
print(f"distance is {distance}")
"""
"""
import numpy as np
gfg = np.random.rayleigh(1, 1)
print(gfg)
"""

'''
ap_num = 5                          # numbers of AP     
    
ap_type = np.dtype([('compu_capa','f4'), ('power', 'f4')]) 
AP = np.zeros(ap_num, dtype= ap_type)  

print(22222222222222)
for ap in AP:
    print(1)
'''

'''
import optimization as op

h=np.array([6.06020304235508*10**-6,1.10331933767028*10**-5,1.00213540309998*10**-7,1.21610610942759*10**-6,1.96138838395145*10**-6,1.71456339592966*10**-6,5.24563569673585*10**-6,5.89530717142197*10**-7,4.07769429231962*10**-6,2.88333185798682*10**-6])
M=np.array([1,0,0,0,1,0,0,0,0,0])

#h = [1,2,3,4,5]

#gain0, off_action = op.cd_method(h, 99)

#print(off_action)
            
# Time resource allocation
gain,a,Tj = op.bisection(h, M, 3, weights=[])

print(f"gain is {gain} a = {a} Tj = {Tj}")
'''




import numpy as np
gfg = np.random.rayleigh(1, 1)
print(gfg)