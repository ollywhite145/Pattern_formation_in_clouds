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
beta=2



qc_eq =( ((d**beta)*(c-a1))/((c**beta)*a2) )**(1/(2*beta -1))
qr_eq = (c*qc_eq)/d

print(qc_eq, qr_eq)

n = 1000
mesh = PeriodicIntervalMesh(n, 50.0)

V = FunctionSpace(mesh, 'CG',1)
W = V*V

qc_= Function(V, name='Cloud')
q1 = Function(V,name='CloudNext')
qr_= Function(V, name='CloudR')
q2 = Function(V,name='CloudRNext')

v1 = TestFunction(V)
v2 = TestFunction(V)


#initial condition set as a gaussian
x = SpatialCoordinate(mesh)  #always returns a vector


pcg = PCG64(seed=123456789)
rg = RandomGenerator(pcg)

qc_pert = rg.normal(V)
qr_pert = rg.normal(V)



q1.assign(qc_eq + 0.1*qc_pert )
q2.assign(qr_eq + 0.1*qr_pert )

#qc.interpolate(b*exp(-(x[0]-20)**2/a))    #again using a gaussian IC  think one of the errors is here...
#qr.interpolate(b*exp(-(x[0]-20)**2/a))       #second variable IC




#q_.assign(q)    #does this need to be put in two parts or is this done automatically
qc_.assign(q1)
qr_.assign(q2)


#qc_,qr_ = q_.split()

 #constants from section 7 in the 1D case

# this timestep doesn't blow up!
#dt = 0.000001
dt=0.1

#n1 = w1*v1*dx
#L1 = ( qc_*v1  + dt*( (c-a1)*qc_*v1 - a2*(qc_**beta)*v1*qr_**beta -D1*inner(grad(qc_),grad(v1)) )  )*dx

#n2= w2*v2*dx
#L2 = ( qr_*v2  + dt*( a1*qc_*v2 - d*qr_*v2 + a2*(qc_**beta)*v2*qr_**beta -D2*inner(grad(qr_),grad(v2)) )  )*dx

#a =n1+n2
#L=L1+L2


F1 = v1*(q1- qc_)*dx - dt*( (c-a1)*q1*v1 - a2*(q1**beta)*v1*q2**beta - D1*inner(grad(q1),grad(v1))  )*dx
F2 = v2*(q2- qr_)*dx - dt*( a1*q1*v2 - d*q2*v2 + a2*(q1**beta)*v2*q2**beta -D2*inner(grad(q2),grad(v2)) )*dx

outfile= File('cloud.pvd')

outfile.write(q1,q2)
#time loop steps

t=0.0
count = 0
end = 200
while (t<=end):
    solve(F1==0, q1)
    qc_.assign(q1)
    solve(F2==0, q2)
    qr_.assign(q2)
    t+=dt
    if count % 10 == 0:
        print("t = ", t)
        outfile.write(q1,q2)
    count += 1
