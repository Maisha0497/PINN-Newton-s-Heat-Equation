# PINN-Newton's-Heat-Equation

![image](https://github.com/Maisha0497/PINN-Newton-s-Heat-Equation/assets/62252679/2f40a8d5-be0e-470b-89a8-43ed5ac34eea)


Here, we show an example on how we used PINN (Physics-Informed Neural Netwwork) when air temperature can heat up via hotter land surface temperatures.
Heating of the air in contact with the land can be considered as a prediction of temperature after a certain time. As discussed previously, this can be done using a combination of physics and Neural Network. Here we can use a combination of loss functions to leverage both the data loss and physics loss which would be similar to given equation. Datasets can be collected from ERA5 or MODIS. The NetCDF ERA5 temperature files have 3D arrays and using extracting values tools we can get point values.
Suppose, we know the land-surface temperature which might be a bit high initially and will cool gradually at a certain rate. Suppose, we know the temperature of the environment and land on a single parcel of area for multiple parcels, that way we can deduce the heating dynamic of the environment using Newton’s heat equation. It goes something like this-
Temperature to be predicted = 
(Air temperature -Temperature of land surface)*e^(-temperature change rate * time elapsed)
+Temperature of land surface; which needs to be differentiated with respect to time.
Using the PyTorch library and torch.autograd module we need to take the partial derivative of the Neural Network itself. We need to figure out the heating rate as the dimension of the land and air are not certain. Thus, we can add the heating rate (r) as a differentiable parameter. 

Fig (heat transfer.png). PINN trained on the temperature data (Source: compiled by author).
We make this PINN using a minimum temperature of 35 celsius and maximum temperature of 45 celsius at a total time lapse of a little over 15 minutes. We start with a heating rate of 0.003 and our network finds a cooling rate of 0.0016 with this PINN.  As we can see there are a few data points and as labelled as training data, in blue dots. What is interesting here is that using only Neural Network regularisation (L2 Network, shown as the red line) can be very deviated from the physical equation. Also, given how the data behaves, there is an increase in temperature at the end after the first 10 minutes. With just the physical equation (orange line), that slight deviation is not evident. Thus, the green line best predicts the temperature change for the land surface after precipitation. 

Thanks to Theodore Wolf, original content at https://medium.com/@theo.wolf/physics-informed-neural-networks-a-simple-tutorial-with-pytorch-f28a890b874a
