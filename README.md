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
* $X_j=\left\{\matrix{1&\text{if plant }j\text{ is open}\\0&\text{otherwise}}\right.$
* $Y_k=\left\{\matrix{1&\text{if DC }k\text{ is open}\\0&\text{otherwise}}\right.$
* $qsp_{ij}=\left\{\matrix{1&\text{if }Qsp_{ij}>0\\0&\text{otherwise}}\right.$
* $qpd_{jk}=\left\{\matrix{1&\text{if }Qpd_{jk}>0\\0&\text{otherwise}}\right.$
* $qdc_{kl}=\left\{\matrix{1&\text{if }Qdc_{kl}>0\\0&\text{otherwise}}\right.$
* $Dsp_{ij}=\left\{\matrix{1&\text{if }fs_i\le fsp_{ij}\\0&\text{otherwise}}\right.$
* $Dpd_{jk}=\left\{\matrix{1&\text{if }Fp_j\le fpd_{jk}\\0&\text{otherwise}}\right.$
* $Ddc_{kl}=\left\{\matrix{1&\text{if }Fd_k\le fdc_{kl}\\0&\text{otherwise}}\right.$

**Constraints**:

$\begin{equation}\end{equation}$
