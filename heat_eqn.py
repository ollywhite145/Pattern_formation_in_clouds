from firedrake import *
import math
import matplotlib.pyplot as plt

a=1
b=1

n = 100
mesh = PeriodicIntervalMesh(n, 40.0)

V = FunctionSpace(mesh, 'CG',1)
V_out=FunctionSpace(mesh, 'CG',1)



q_= Function(V, name='Heat')   # same as Burgers but with q instead of u
q = Function(V,name='HeatNext')

w = TrialFunction(V)
v = TestFunction(V)



#initial condition set as a gaussian
x = SpatialCoordinate(mesh)

#ic = project(b*exp(-x[0]**2/a),V)  #this is the equivalent as 
#the bugers equation so not sure if I should use this or the line below

q.interpolate(b*exp(-x**2/a))   # gaussian function be^(-x^2/a)



q_.assign(q)

D = 0.01
dt=0.1


a = inner(w,v)*dx
L = ( inner(q_,v )- dt*inner(grad(q_),grad(v)) )*dx








outfile= File('Heat_eqn.pvd')
outfile.write(project(q, V_out, name="Heat"))


#time loop steps

t=0.0
end = 10
while (t<=end):
    solve(a==L,q)
    q_.assign(q)
    t+=dt
    outfile.write(project(q,V_out,name='Heat'))
