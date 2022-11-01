# R dial to N Values
Measurements of absorption are collated in $R$-to-$N$ table [doc 3]. Each wavelength pair (A, C and D) has its own R-to-N table, given as absorption at 0, 10, .. 300 degrees, obtained from the wedge of the [Dobson Spectrophotometer](https://www.jma-net.go.jp/kousou/obs_third_div/ozone/ozone_dob-e.html) 

* To obtain values of $N$ at different degrees, use linear interpolation. 
* Negative absorption means absorption below calibration value


## Instrument constant
This is a representation of error, due to bad calibration of spectrophotometer. The values in $R$-to-$N$ table are subject to calibration error, which must be estimated as a constant $K$ for the given wavelength
$$L_0=\log{\frac{I_0}{I_0^\prime}}+k$$
This leads to corresponding error $S$
$$N = N^*+S$$
where $N^*$ is from the R-to-N table.

Note: 
1. Each wavelength, A, C and D, has its own error calibration errro $S$. The wavelength pairs have its own $S$. 
2. Since $X$ should remain constant within 24-hour intervals, the true $S$ should minimise the spread of $X$, i.e. the mean std of $X$, averaged over the season. 
	![[Pasted image 20220722173531.png|350]] 

### Langley method: 1 day, 1 wavelngth, 3 measurements
1. In expression for $X$ ([[1 Direct Sun Equation#^c55034]]), set $N:=N^*+S$, $\Delta\delta=0$ (true for Antarctica), $p=1$, $m=\mu$. We then obtain the following:
$$\frac{N^*}{\mu}=-\frac{S}{\mu}+(X\Delta\alpha +\Delta\beta)$$
2. We have several measurements (e.g., three) of $N^*$ and $\mu$, performed on the same day where $(X\Delta\alpha +\Delta\beta)=const$ and $S=const$, but $N^*$ and $\mu$ differ between the measurements.
3. Estimate $S$ by doing a linear fit on $N^*/\mu$ vs $1/\mu$. For A, C, D wavelengths an example is in [doc 2]:
	![[Pasted image 20220722165610.png|350]]
4. Potentially, average fitted $S$ across more days. 

