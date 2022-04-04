#First attempt of the cloud model

from firedrake import *
import math
import matplotlib.pyplot as plt

a=1
b=1

a1=1
a2=1
c=5
d=0.1
D1=1000
D2=0.1
beta=1



qc_eq =( ((d**beta)*(c-a1))/((c**beta)*a2) )**(1/(2*beta -1))
qr_eq = (c*qc_eq)/d


n = 100
mesh = PeriodicIntervalMesh(n, 40.0)

V = FunctionSpace(mesh, 'CG',1)
W = V*V

q_= Function(W, name='Cloud')
q = Function(W,name='CloudNext')

w1,w2 = TrialFunctions(W)
v1,v2 = TestFunctions(W)


#initial condition set as a gaussian
x = SpatialCoordinate(mesh)  #always returns a vector


pcg = PCG64(seed=123456789)
rg = RandomGenerator(pcg)

qc_pert = rg.normal(V)
qr_pert = rg.normal(V)



qc,qr = q.split()

qc.assign(qc_eq+0.01*qc_pert )
qr.assign(qr_eq+0.01*qr_pert )

#qc.interpolate(b*exp(-(x[0]-20)**2/a))    #again using a gaussian IC  think one of the errors is here...
#qr.interpolate(b*exp(-(x[0]-20)**2/a))       #second variable IC




q_.assign(q)    #does this need to be put in two parts or is this done automatically
#qc_.assign(qc)
#qr_.assign(qr)


qc_,qr_ = q_.split()

 #constants from section 7 in the 1D case

dt=0.005


n1 = inner(w1,v1)*dx
L1 = ( inner(qc_,v1)  + dt*( (c-a1)*inner(qc_,v1) - a2*inner(qc_**beta,v1)*qr_**beta -D1*inner(grad(qc_),grad(v1)) )  )*dx

n2= inner(w2,v2)*dx
L2 = ( inner(qr_,v2)  + dt*( a1*inner(qc_,v2) - d*qr_*v2 + a2*inner(qc_**beta,v2)*qr_**beta -D1*inner(grad(qr_),grad(v2)) )  )*dx


a =n1+n2
L=L1+L2


outfile= File('cloud.pvd')	

outfile.write(qc,qr)
#time loop steps

t=0.0
end = 1
while (t<=end):
    solve(a==L,q)
    q_.assign(q)
    t+=dt
    qc,qr = q.split()
    outfile.write(qc,qr)
