# Genetic Algorithm of Supply Chian Resilience

Source code of developing a reliable and flexible supply chain network design model using genetic algorithm.

## Code Structure

### Files (src)

* instance_generator.py
* solver.py
* boundary_generator.py
* genetic.py
* solve.py
* list_tools.py

### Explanations

#### instance_generator.py

Generate instances with different scales (20 scales totallyl) according to some uniform distribution.

All datas follow the following distribution:

![](https://cdn.jsdelivr.net/gh/TseYeong/blog-plug@main/image-20250106194122506.png)

And the number of facilities of each instance is shown in the following:

![](https://cdn.jsdelivr.net/gh/TseYeong/blog-plug@main/image-20250106200700963.png)

**Output**: the generated data is written to the absolute path `./instances/Problem code/`

#### solver.py

This script creates an optimization model on cost, reliability or flexibility based on Gurobi, respectively.

Full description of **parameters** of the optimal problem is:

* $FCp_j$	Fixed cost of opening a plant $j$

* $FCd_k$	Fixed cost of opening a DC $k$.
* $Ssp_{ij}$	Unit shipping cost from supplier $i$ to plant $j$.
* $Spd_{jk}$	Unit shipping cost from plant $j$ to DC $k$.
* $Sdc_{kl}$	Unit shipping cost from DC $k$ to CZ $l$.
* $Cs_i$	Capacity of supplier $i$.
* $Cp_j$	Capcity of plant $j$.
* $Cd_k$	Capacity of DC $k$.
* $D_l$	Demand ot customer zone $l$.
* $SL_l$	Fill rate (Service level) target at customer zone $l$.
* $rs_i$	Reliability index of supplier $i$.
* $rp_j$	Reliability index of plant $j$.
* $rd_k$	Reliability idnex of DC $k$.
* $rc_l$	Reliability index of CZ $l$.
* $rsp_{ij}$	Reliability index of link between supplier $i$ and plant $j$.
* $rpd_{jk}$	Reliability index of link between plant $j$ and DC $k$.
* $rdc_{kl}$	Reliability index of link between DC $k$ and CZ $l$.
* $fs_i$	Volume flexibility index of supplier $i$.
* $fsp_{ij}$	Volume flexibility index of link between supplier $i$ and plant $j$.
* $fpd_{jk}$	Volume flexibility index of link between plant $j$ and DC $k$.
* $fdc_{kl}$	Volume flexibility index of link between DC $k$ and CZ $l$.

**Decision variables** is given by following:

* $Qsp_{ij}$	Quantity of items shipped from supplier $i$ to plant $j$.
* $Qpd_{jk}$	Quantity of items shipped from plant $j$ to DC $k$.
* $Qdc_{kl}$	Quantity of items shipped from DC $k$ to CZ $l$.
* $Rp_j$	Cumulative reliability index at plant $j$.
* $Rd_k$	Cumulative reliability index at DC $k$.
* $Rc_l$	Cumulative reliability index at CZ $l$.
* $Fp_j$	Cumulative volume flexibility index at plant $j$.
* $Fd_k$	Cumulative volume flexibility index at DC $k$.
* $Fc_l$	Cumulative volume flexibility index at CZ $l$.
* <img src="https://latex.codecogs.com/svg.image?X_j=\begin{cases}1&\text{if plant}\; j\;\text{is open}\\ 0&\text{otherwise}\end{cases}" /> 
* <img src="https://latex.codecogs.com/svg.image?Y_k=\begin{cases}1&\text{if&space;DC}\;k\;\text{is&space;open}\\0&\text{otherwise}\end{cases}" title="Y_k=\begin{cases}1&\text{if DC}\;k\;\text{is open}\\0&\text{otherwise}\end{cases}" /> 
* <img src="https://latex.codecogs.com/svg.image?qsp_{ij}=\begin{cases}1&\text{if}\;Qsp_{ij}>0\\0&\text{otherwise}\end{cases}" title="qsp_{ij}=\begin{cases}1&\text{if}\;Qsp_{ij}>0\\0&\text{otherwise}\end{cases}" /> 
* <img src="https://latex.codecogs.com/svg.image?qpd_{jk}=\begin{cases}1&\text{if}\;Qpd_{jk}>0\\0&\text{otherwise}\end{cases}" title="qpd_{jk}=\begin{cases}1&\text{if}\;Qpd_{jk}>0\\0&\text{otherwise}\end{cases}" /> 
* <img src="https://latex.codecogs.com/svg.image?qdc_{kl}=\begin{cases}1&\text{if}\;Qdc_{kl}>0\\0&\text{otherwise}\end{cases}" title="qdc_{kl}=\begin{cases}1&\text{if}\;Qdc_{kl}>0\\0&\text{otherwise}\end{cases}" /> 
* <img src="https://latex.codecogs.com/svg.image?Dsp_{ij}=\begin{cases}1&\text{if}\;fs_i\le&space;fsp_{ij}\\0&\text{otherwise}\end{cases}" title="Dsp_{ij}=\begin{cases}1&\text{if}\;fs_i\le fsp_{ij}\\0&\text{otherwise}\end{cases}" /> 
* <img src="https://latex.codecogs.com/svg.image?Dpd_{jk}=\begin{cases}1&\text{if}\;Fp_j\le&space;fpd_{jk}\\0&\text{otherwise}\end{cases}" title="Dpd_{jk}=\begin{cases}1&\text{if}\;Fp_j\le fpd_{jk}\\0&\text{otherwise}\end{cases}" /> 
* <img src="https://latex.codecogs.com/svg.image?Ddc_{kl}=\begin{cases}1&\text{if}\;Fd_k\le&space;fdc_{kl}\\0&\text{otherwise}\end{cases}" title="Ddc_{kl}=\begin{cases}1&\text{if}\;Fd_k\le fdc_{kl}\\0&\text{otherwise}\end{cases}" /> 

**Constraints**:

* $\displaystyle\sum_{j=1}^{J}Qsp_{ij}\le Cs_i\quad\forall i$ 
* $\displaystyle\sum_{k=1}^{K}Qpd_{jk}\le Cp_jX_j\quad\forall j$ 
* $\displaystyle\sum_{l=1}^{L}Qdc_{kl}\le Cd_kY_k\quad\forall k$
* $\displaystyle\sum_{k=1}^KQdc_{kl}\ge D_lSL_l\quad \forall l$
* $\displaystyle\sum_{i=1}^{I}Qsp_{ij}=\sum_{k=1}^{K}Qpd_{jk}\quad\forall j$
* $\displaystyle\sum_{j=1}^{J}Qpd_{jk}=\sum_{l=1}^{L}Qdc_{kl}\quad\forall k$
* $\displaystyle Rp_j=rp_j\left[1-\prod_{i=1}^{I}(1-qsp_{ij}rsp_{ij}rs_i)\right]\quad\forall j$
* $\displaystyle Rd_k=rd_k\left[1-\prod_{j=1}^{J}(1-qpd_{jk}rpd_{jk}Rp_j)\right]\quad\forall k$
* $\displaystyle Rc_l=\left[1-\prod_{k=1}^{K}(1-qdc_{kl}rdc_{kl}Rd_k)\right]\quad\forall l$
* $\displaystyle Fsp_{ij}=Dsp_{ij}fs_i+(1-Dsp_{ij})fsp_{ij}\quad\forall i,j$
* $\displaystyle Fp_{j}=\frac{\sum_{i=1}^{I}Qsp_{ij}Fsp_{ij}}{\sum_{i=1}^{I}Qsp_{ij}}\quad\forall j$
* $\displaystyle Fpd_{jk}=Dpd_{jk}Fp_j+(1-Dpd_{jk})fpd_{jk}\quad\forall j,k$
* $\displaystyle Fd_k=\frac{\sum_{j=1}^{J}Qpd_{jk}Fpd_{jk}}{\sum_{j=1}^{J}Qpd_{jk}}\quad\forall k$
* $\displaystyle Fdc_{kl}=Ddc_{kl}Fd_k+(1-Ddc_{kl})fdc_{kl}\quad\forall k,l$
* $\displaystyle Fc_l=\frac{\sum_{k=1}^{K}Qdc_{kl}Fdc_{kl}}{\sum_{k=1}^{k}Qdc_{kl}}\quad\forall l$
* $\displaystyle\sum_{k=1}^{K}qdc_{kl}=1$
* $Qsp_{ij}\ge 0\quad\forall i,j$
* $Qpd_{jk}\ge 0\quad\forall j,k$
* $Qdc_{kl}\ge 0\quad\forall k,l$

**Objectives**:

Objective 1: ***cost minimization***

<img src="https://latex.codecogs.com/svg.image?\inline&space;\large&space;\begin{equation}\begin{split}\min&space;Z_1=&\sum_{j=1}^{J}FCp_jX_j&plus;\sum_{k=1}^{K}FCd_kY_k\\&&plus;\sum_{i=1}^{I}\sum_{j=1}^{J}Ssp_{ij}Qsp_{ij}&plus;\sum_{j=1}^{J}\sum_{k=1}^{K}Spd_{jk}Qpd_{jk}\\&&plus;\sum_{k=1}^{K}\sum_{l=1}^{L}Sdc_{kl}Qdc_{kl}\end{split}\end{equation}" title="\begin{equation}\begin{split}\min Z_1=&\sum_{j=1}^{J}FCp_jX_j+\sum_{k=1}^{K}FCd_kY_k\\&+\sum_{i=1}^{I}\sum_{j=1}^{J}Ssp_{ij}Qsp_{ij}+\sum_{j=1}^{J}\sum_{k=1}^{K}Spd_{jk}Qpd_{jk}\\&+\sum_{k=1}^{K}\sum_{l=1}^{L}Sdc_{kl}Qdc_{kl}\end{split}\end{equation}" />

Objective 2: ***maximization of network reliability***

<img src="https://latex.codecogs.com/svg.image?\inline&space;\large&space;\displaystyle\max&space;Z_2=\sum_{l=1}^{L}Wr_lRc_l" title="\displaystyle\max Z_2=\sum_{l=1}^{L}Wr_lRc_l" />

Here $Wr_l$ is relative weight allocated for reliability requirement at customer zone $l$, we simply consider $Wr_l=1/L,\forall l$ here.

Objective 3: ***maximization of network flexibility***

<img src="https://latex.codecogs.com/svg.image?\inline&space;\large&space;\displaystyle\max&space;Z_3=\sum_{l=1}^{L}Wf_lFc_l" title="\displaystyle\max Z_3=\sum_{l=1}^{L}Wf_lFc_l" />

Same as reliability, we consider $Wf_l=1/L, \forall l$ here.

**Output**:

A optimization model with specific objective ($Z_1,Z_2\text{ or }Z_3$).

#### boundary_generator.py

This script generate the boundary (i.e., max and min value) of each objective and save the result as a local file.

**Notice**: You need to run this script with Gurobi license, or you could only run this script on small and medium problem types. For more information, visit the official website of Gurobi [here](https://support.gurobi.com/hc/en-us/sections/360001969411-Accounts-and-Licensing)

#### genetic.py

The implementation of genetic algorithm on current supply chain oprimization problem.

**Fitness value**: calculated by normalizing each objective, the following formula is used to normalize the objectives:

<img src="https://latex.codecogs.com/svg.image?\inline&space;\large&space;\displaystyle&space;Z_1^{norm}=\frac{\textbf{Max}(Z_1)-Z_1}{\textbf{Max}(Z_1)-\textbf{Min}(Z_1)}" title="\displaystyle Z_1^{norm}=\frac{\textbf{Max}(Z_1)-Z_1}{\textbf{Max}(Z_1)-\textbf{Min}(Z_1)}" />

<img src="https://latex.codecogs.com/svg.image?\inline&space;\large&space;\displaystyle&space;Z_{2,3}^{norm}=\frac{Z_{2,3}-\textbf{Min}(Z_{2,3})}{\textbf{Max}(Z_{2,3})-\textbf{Min}(Z_{2,3})}" title="\displaystyle Z_{2,3}^{norm}=\frac{Z_{2,3}-\textbf{Min}(Z_{2,3})}{\textbf{Max}(Z_{2,3})-\textbf{Min}(Z_{2,3})}" />

Then fitness value is obtained by:

<img src="https://latex.codecogs.com/svg.image?\inline&space;\large&space;\displaystyle\sum_{i=1}^3w_iZ_i^{norm}" title="\displaystyle\sum_{i=1}^3w_iZ_i^{norm}" />

where, 

<img src="https://latex.codecogs.com/svg.image?\inline&space;\large&space;\displaystyle\sum_{i=1}^3w_i=1,\text{and}\;w_i\ge&space;0\quad\forall&space;i" title="\displaystyle\sum_{i=1}^3w_i=1,\text{and}\;w_i\ge 0\quad\forall i" />

Our goal now is to maximize the fitness value.

