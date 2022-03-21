#Fisher equation python file

from firedrake import *
import math
import matplotlib.pyplot as plt

a=1
b=1


k=1    # fisher equation is du/dt = ku(1-u)+D d^2u/dx^2
 


n = 100
mesh = PeriodicIntervalMesh(n, 40.0)

V = FunctionSpace(mesh, 'CG',1)
V_out=FunctionSpace(mesh, 'CG',1)



q_= Function(V, name='Fisher')
q = Function(V,name='FisherNext')

w = TrialFunction(V)
v = TestFunction(V)



#initial condition set as a gaussian
x = SpatialCoordinate(mesh)

#ic = project(b*exp(-x[0]**2/a),V)  #this is the equivalent as 
#the bugers equation so not sure if I should use this or the line below

q.interpolate(b*exp(-(x[0]-20)**2/a))   # gaussian function but could have somethign else?


q_.assign(q)

D = 0.01
dt=0.1


a = inner(w,v)*dx  #same LHS as heat equation
L = ( inner(q_,v )- dt*(k*inner(q_,v)  -k*inner(inner(q_,q_),v) + D*inner(grad(q_),grad(v))) )*dx



try:

	outfile= File('Fisher.pvd')
	outfile.write(project(q, V_out, name="fisher"))
except Exception as e:
	print(e)
#time loop steps

t=0.0
end = 10
while (t<=end):
    solve(a==L,q)
    q_.assign(q)
    t+=dt
    outfile.write(project(q,V_out,name='fisher'))
