from firedrake import *
import math
import matplotlib.pyplot as plt

#constants in the 1D case
a1=1
a2=1
c=-5
d=0.1
D1=1000
D2=0.1
beta1=2
beta2=1
B=0.15

#calculates the equilibrium value.
qc_eq =( ((d**beta1)*(c-a1))/((c**beta1)*a2) )**(1/(2*beta1 -1))
qr_eq = (c*qc_eq)/d
print(qc_eq, qr_eq)

n = 1000 #number of points in the mesh.
mesh = PeriodicIntervalMesh(n, 50.0) #a mesh of length 50 with 1000 points


V = FunctionSpace(mesh, 'CG',1)
W = V*V

#Defining the variables required for the timestepping scheme.
qc_= Function(V, name='Cloud')
q1 = Function(V,name='q_c')
qr_= Function(V, name='CloudR')
q2 = Function(V,name='q_r')

#the test functions that multiply the equations to return the weak form.
v1 = TestFunction(V)
v2 = TestFunction(V)

x = SpatialCoordinate(mesh)  #defining the spatial coordinate based on the mesh defined above.

#generating the random pertubation to apply to the initial condition.
pcg = PCG64(seed=123456789)
rg = RandomGenerator(pcg)

qc_pert = rg.normal(V)
qr_pert = rg.normal(V)

#equilibrium state plus a small random pertubation.
q1.assign(qc_eq + 0.1*qc_pert )
q2.assign(qr_eq + 0.1*qr_pert )


qc_.assign(q1)
qr_.assign(q2)

#timestep.
dt=0.1

#the discretized cloud model equations.
F1 = v1*(q1- qc_)*dx - dt*( (c-a1)*(q1**beta1)*v1 - a2*(q1**beta2)*v1*(q2**beta2) - D1*inner(grad(q1),grad(v1))  )*dx
F2 = v2*(q2- qr_)*dx - dt*(B*v2+ a1*(q1**beta1)*v2 - d*q2*v2 + a2*(q1**beta2)*v2*(q2**beta2) -D2*inner(grad(q2),grad(v2)) )*dx

#defining the file where the results are written to.
outfile= File('play.pvd')
outfile.write(q1,q2)

#time loop steps
t=0.0
count = 0
end = 2000
while (t<=end):
    solve(F1==0, q1) #solving the cloud scheme for q_c.
    qc_.assign(q1)
    solve(F2==0, q2) #solving the cloud scheme for q_r.
    qr_.assign(q2)
    t+=dt
    if count % 100 == 0: #reducing the number of files created by only creating a file every 100 timesteps
        print("t = ", t)
        outfile.write(q1,q2)
    count += 1
