from qiskit import * 
import math
import random
import numpy as np
from qiskit import QuantumCircuit, execute, Aer
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize

#generate ansatz
        
def apply_fixed_ansatz(qubits,layer,parameters):
    
    qc = QuantumCircuit(qubits)

    for i in range(qubits):
        qc.h(i)
        
    for l in range(layer) :
        for i in range(qubits):
            qc.ry(parameters[2*l*qubits+i],i)
        for i in range(int(qubits/2)):
            qc.cz(2*i,2*i+1)
        for i in range(qubits):
            qc.ry(parameters[(2*l+1)*qubits+i],i)
        for i in range(int(qubits/2-1/2)):
            qc.cz(2*i+1,2*i+2)
    return qc

def objective_function(parameters): 
    
    circ1 = apply_fixed_ansatz(qubits,layer,parameters)
    circ1.h([3])
    circ1.h([2])
    circ1.h([1])
    circ1.h([0])
    circ1.measure_all()
    shots = 1000
    backend = BasicAer.get_backend('qasm_simulator')
    result1 = backend.run(transpile(circ1, backend), seed_simulator=10,shots=shots).result()
    counts1 = result1.get_counts(circ1)
    bit_string1 = counts1.keys()
    qubit_mea1 = [0]
    E1 = 0
    for s in bit_string1:
        s1=''                          
        if len(qubit_mea1)!=0:
            for qm in qubit_mea1:
                s1+=s[::-1][qm]
        if s1== '0':
            E1+=counts1[s]/shots 
        elif s1 == '1':
            E1-=counts1[s]/shots
   
    qubit_mea11 = [0,1]
    E11 = 0
    for s in bit_string1:
        s11=''                          
        if len(qubit_mea11)!=0:
            for qm in qubit_mea11:
                s11+=s[::-1][qm]
        if s11== '00' or s11 == '11':
            E11+=counts1[s]/shots 
        elif s11 == '01' or s11 == '10':
            E11-=counts1[s]/shots
    
    qubit_mea2 = [2]
    E2 = 0
    for s in bit_string1:
        s2=''                          
        if len(qubit_mea2)!=0:
            for qm in qubit_mea2:
                s2+=s[::-1][qm]
        if s2== '0' :
            E2+=counts1[s]/shots 
        elif s2 == '1':
            E2-=counts1[s]/shots
       
    qubit_mea21 = [2,3]
    E21 = 0
    for s in bit_string1:
        s21=''                          
        if len(qubit_mea21)!=0:
            for qm in qubit_mea21:
                s21+=s[::-1][qm]
        if s21== '00' or s21 == '11' :
            E21+=counts1[s]/shots 
        elif s21== '01' or s21 == '10':
            E21-=counts1[s]/shots
            
    circ2 = apply_fixed_ansatz(qubits,layer,parameters) 
    circ2.rx(np.pi/2, [3])
    circ2.rx(np.pi/2, [2])
    circ2.rx(np.pi/2, [1])
    circ2.rx(np.pi/2, [0])
    circ2.measure_all()
    result2 = backend.run(transpile(circ2, backend), seed_simulator=10,shots=shots).result()
    counts2 = result2.get_counts(circ2)
    bit_string2 = counts2.keys()
    qubit_mea2 = [0,1]
    E0 = 0
    for s in bit_string2:
        s0=''                          
        if len(qubit_mea2)!=0:
            for qm in qubit_mea2:
                s0+=s[::-1][qm]
        if s0 == '00' or s0 == '11':
            E0+=counts2[s]/shots 
        elif s0 == '01' or s0 == '10':
            E0-=counts2[s]/shots
            
    qubit_mea22 = [2,3]
    E01 = 0
    for s in bit_string2:
        s01=''                          
        if len(qubit_mea22)!=0:
            for qm in qubit_mea22:
                s01+=s[::-1][qm]
        if s01 == '00' or s01 == '11':
            E01+=counts2[s]/shots 
        elif s01 == '01' or s01 == '10':
            E01-=counts2[s]/shots
    
    circ7 = apply_fixed_ansatz(qubits,layer,parameters)
    a = Statevector.from_instruction(circ7)
    g = a.data
    j= np.array([0.1381966, 0.2236068, 0.2236068, 0.1381966, 0.2236068, 0.3618034, 0.3618034, 0.2236068, 0.2236068, 0.3618034, 0.3618034, 0.2236068, 0.1381966, 0.2236068,
 0.2236068, 0.1381966])
    
    prob = np.dot(g,j) **2
   
        

    return  -0.125 *prob*(1+0.25*(E1+E2+0.5*(E11+E21+E0+E01)))   
   

# 设置初始参数值

def Gradient(theta):
    g = np.zeros(2 * layer * qubits) 
    for i in range(2 * layer * qubits): 
        theta_temp = theta.copy()
        theta_temp[i]+= np.pi/4
        e1 = objective_function(theta_temp) 
        #print(i,theta)
        theta_temp[i]-= np.pi/2
        e2 = objective_function(theta_temp)
        theta_temp[i]+= np.pi/4
        g[i] = e1 .real - e2.real
    return g


def adam_iters(theta0,max_iter):
    final_precision = 1e-4
    theta_precision = 1e-2
    beta1 = 0.9
    beta2 = 0.999
    alpha = 0.1
    epsilon = 10 **(-8)

    npoints = 2* layer * qubits
    

    #adam gradient descent algorithm

    m=np.zeros(npoints,dtype=np.float64)
    v=np.zeros(npoints,dtype=np.float64)   

    theta=np.zeros(shape=(int(max_iter+1),npoints),dtype=np.float64)
    energy =np.zeros((int(max_iter+1)),dtype=np.float64)
    g=np.zeros(shape=(int(max_iter+1),npoints),dtype=np.float64)

    theta[0]= theta0.copy()
    
    
    g[0] = Gradient(theta[0])
    energy[0] = objective_function(theta[0]).real
    

    
    t=0
    for t in range(1,max_iter+1):
        m=beta1*m+(1-beta1)*g[t-1]
        v=beta2*v+(1-beta2)*g[t-1]**2
        
        alphat=alpha*np.sqrt(1-beta2**t)/(1-beta1**t)
        theta_change=alphat*m/(np.sqrt(v)+epsilon)
        theta[t]=theta[t-1]-theta_change
        
        g[t]=Gradient(theta[t])
        energy[t] = objective_function(theta[t]).real
          

        if np.abs(energy[t]-energy[t-1]) < final_precision and np.linalg.norm(theta_change)<theta_precision:
            break   
            
            
        if t%10==0:
            print(t)
            
    return  theta[:t], energy[:t]

qubits=4
layer=2
max_iter=100
np.random.seed(10)
theta0=np.random.uniform(-np.pi/2,np.pi/2,2 * layer * qubits)
opt_para1, e1 = adam_iters(theta0,max_iter)
opt_para = opt_para1[-1]
e = e1[-1]

qc = apply_fixed_ansatz(qubits, layer, opt_para)
d = Statevector.from_instruction(qc)
b = d.data

x=np.array( [0.1381966, 0.2236068, 0.2236068, 0.1381966,
 0.2236068, 0.3618034 ,0.3618034, 0.2236068,
 0.2236068, 0.3618034, 0.3618034, 0.2236068,
 0.1381966, 0.2236068, 0.2236068, 0.1381966])

f=np.abs(b.dot(x))
print(f,e,opt_para,b)
j= np.array([0.1381966, 0.2236068, 0.2236068, 0.1381966, 0.2236068, 0.3618034, 0.3618034, 0.2236068, 0.2236068, 0.3618034, 0.3618034, 0.2236068, 0.1381966, 0.2236068,
 0.2236068, 0.1381966])
r=np.dot(j,b)
print('r',r)
