from firedrake import *
import math
import matplotlib.pyplot as plt

#constants from section 7 in the 2D case
a1=1
a2=1
c=5
d=0.13    #note different to the 1D
D1=100    #note different to the 1D
D2=0.025  #note different to the 1D
beta=2


qc_eq =( ((d**beta)*(c-a1))/((c**beta)*a2) )**(1/(2*beta -1)) #calculates the equilibrium value.
qr_eq = (c*qc_eq)/d

print(qc_eq, qr_eq)  #checking that the equilibrium state is calculated correctly.


n = 1000 #number of points in the mesh.
mesh = PeriodicSquareMesh(n,n, 50.0)   #a 50x50 periodic mesh.

V = FunctionSpace(mesh, 'CG',1)
W = V*V

#Defining the variables required for the timestepping scheme.
qc_= Function(V, name='Cloud')
q1 = Function(V,name='CloudNext')
qr_= Function(V, name='CloudR')
q2 = Function(V,name='CloudRNext')


#the test functions that multiply the equations to return the weak form.
v1 = TestFunction(V)
v2 = TestFunction(V)

#defining the spatial coordinate based on the mesh defined above.
x = SpatialCoordinate(mesh)

#generating the random pertubation to apply to the initial condition.
pcg = PCG64(seed=123456789)
rg = RandomGenerator(pcg)

qc_pert = rg.normal(V)
qr_pert = rg.normal(V)

q1.assign(qc_eq + 0.1*qc_pert )  #equilibrium state plus a small random pertubation.
q2.assign(qr_eq + 0.1*qr_pert )


qc_.assign(q1)
qr_.assign(q2)

#timestep.
dt=0.1

#the discretized cloud model equations.
F1 = v1*(q1 - qc_)*dx - dt*( (c-a1)*q1*v1 - a2*(q1**beta)*v1*q2**beta - D1*inner(grad(q1),grad(v1))  )*dx
F2 = v2*(q2 - qr_)*dx - dt*( a1*q1*v2 - d*q2*v2 + a2*(q1**beta)*v2*q2**beta -D2*inner(grad(q2),grad(v2)) )*dx

#defining the file where the results are written to.
outfile= File('cloud2D.pvd')
outfile.write(q1,q2)

#time loop steps.
t=0.0
count = 0
end = 120
while (t<=end):
    solve(F1==0, q1) #solving the cloud scheme for q_c.
    qc_.assign(q1)
    solve(F2==0, q2) #solving the cloud scheme for q_r.
    qr_.assign(q2)
    t+=dt
    if count % 20 == 0: #reducing the number of files created by only creating a file every 20 timesteps
        print("t = ", t)
        outfile.write(q1,q2)
    count +=1
