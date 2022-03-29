#First attempt of the cloud model

from firedrake import *
import math
import matplotlib.pyplot as plt

a=1
b=1


n = 100
mesh = PeriodicIntervalMesh(n, 40.0)

V = FunctionSpace(mesh, 'CG',1)
W = V*V

q_= Function(W, name='Cloud')
q = Function(W,name='CloudNext')

w1,w2 = TrialFunctions(W)
v1,v2 = TestFunctions(W)


#initial condition set as a gaussian
x = SpatialCoordinate(mesh)


qc,qr = q.split()


qc.interpolate(b*exp(-(x[0]-20)**2/a))    #again using a gaussian IC
qr.interpolate(b*exp(-(x[0]-20)**2/a)       #second variable IC

q_.assign(q)    #does this need to be put in two parts or is this done automatically
#qc_.assign(qc)
#qr_.assign(qr)


D1 = 1000
D2 = 0.1

a1=1
a2=1
c=5
d=0.1  #constants from section 7 in the 1D case

dt=0.1


n1 = inner(w1,v1)*dx
L1 = ( inner(qc_,v1)  + dt*( (c-a1)*inner(qc_,v1) - a2*inner(qc_**2,v1)*qr_**2 -D1*inner(grad(qc_),grad(v1)) )  )*dx

n2
L2 = ( inner(qr_,v2)  + dt*( a1*inner(qr_,v2) + a2*inner(qc_**2,v2)*qr_**2 -D1*inner(grad(qr_),grad(v2)) )  )*dx

try:

	outfile= File('cloud.pvd')
	outfile.write(q)
except Exception as e:
	print(e)
#time loop steps

t=0.0
end = 10
while (t<=end):
    solve(n1==L1,q)
    solve(n2==L2,q)
    q_.assign(q)
    t+=dt
    outfile.write(project(q,V_out,name='cloud'))
