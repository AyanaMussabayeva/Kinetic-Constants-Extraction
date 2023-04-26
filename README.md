# Kinetic-Constants-Extraction

Catalytic chemical reactions are complex process, which can be represented as a system of ordinary differential equations (ODEs) by deriving the rate laws for each of the species in this reaction. These systems of ODEs can refer to different kinetic mechanisms. 

Kinetic mechanism shows the dependency of the reaction's rate with the kinetic constants and species' concentrations.  Analyzing the kinetics of a reaction is crucial for understanding the behaviour of the reaction and for improving its performance. Chemists would like to automate this process by enabling extraction of the kinetics from time-series concentrations data. 

![Alt text](/figures/general_idea.png "General idea") 

The problem is formulated as a least squares problem.  Each ODE of the system can be written as: 
$\dot{X} = \widetilde{X}W$

![Alt text](/figures/main_scheme.png "General scheme for ODEs extraction from time-series data")

The data used for the experiments is generated numerically using fourth order Runge-Kutta method. Different data sampling is used to check how the distribution of components concentration's data affect the final result. Three sampling options are used in the experiments: equispaced sampling, points adaptively chosen by Runge-Kutta-Felhberg 4(5) and Chebyshev points.

Apart from receiving a solution that fits the data trajectories, it is important to obtain sparse solution. This can be achieved by using iterative thresholding algorithms or regularization techniques. Thus, sequentially thresholded least squares (STLSQ) algorithm and Lasso regularization have been used to extract interpretable system of ODEs from the generated data trajectories. 
  
The obtained results show that Lasso regularization is more stable to numerical error and is able to identify the correct ``active" components (those, which should have non-zero coefficients in the right hand side of the ODE) more often that STLSQ. It also has been seen that the result is significantly affected by the choice of numerical approximation for derivatives calculation and data sampling. 
