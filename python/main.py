import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from juliacall import Main as jl
import time

#Load in my julia methods
jl.seval('include("julia/methods.jl")')
jl.seval('include("julia/functions.jl")')
#Save the data from Julia calls to Python into variables
exp_eul_data = jl.euler(jl.exp_decay, 1.0, 0.0, 5.0, 0.01)



times = np.asarray(exp_eul_data.times)
states = np.asarray(exp_eul_data.states)
derivatives = np.asarray(exp_eul_data.derivatives) 


#Creating and saving my column vectors for x and y from Julia
x_train = torch.tensor(
    states, dtype=torch.float32).reshape(-1, 1)
y_train = torch.tensor(
    derivatives, dtype=torch.float32).reshape(-1, 1)

#Makes our neural network, goes in order of operations from top to bottom
model = nn.Sequential(
    nn.Linear(1, 16), #Takes in our input data vector
    nn.Tanh(), #Applies the nonlinear activation
    nn.Linear(16, 1) #Pulls from activated neurons to make our predicted output vector
)
#Loss function, mean squared error
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

loss_history = [] 
start_time = time.perf_counter()
# Train the neural network
for epoch in range(2000):
    predictions = model(x_train) 
    #Takes in our input data, runs through nn, saves to predictions
    loss = loss_fn(predictions, y_train)
    #Calculates the loss function
    loss_history.append(loss.item()) 
    #Saves our current loss
    optimizer.zero_grad()
    #Cleans out the current gradient of loss function for current parameters
    loss.backward()
    #Takes our loss function and computes a gradient
    optimizer.step()
    #Updates the parameter vector by running Adam based on our gradients

elapsed_time = time.perf_counter() - start_time
best_iteration = int(np.argmin(loss_history)) + 1
best_loss = min(loss_history)

print("Best loss:", best_loss)
print("Best iteration:", best_iteration)
print("Final loss:", loss_history[-1])
print("Training time (seconds):", elapsed_time)
print("Iterations:", len(loss_history))

plt.figure()
plt.plot(range(1, len(loss_history) + 1), loss_history)
plt.xlabel("Iteration")
plt.ylabel("Mean squared error")
plt.title("Training Loss vs. Iteration")
plt.yscale("log")
plt.grid(True)
plt.show()