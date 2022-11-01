# Direct Sun Equation
Central quantitiy is Ozone amount in Dobson units, $X$. This is obtained from measurements of absorption, performed by [Dobson Spectrophotometer](https://www.jma-net.go.jp/kousou/obs_third_div/ozone/ozone_dob-e.html)

The measurements are done at three different wavelength pairs, denoted A, C and D.

The shorter wavelengths are more strongly absorbed by Ozone than the longer ones. 

## Single wavelength pair equation
Given a wavelength pair with a short and long wavelengths, all deltas ($\Delta$-quantities) are w.r.t. to short and long:

$$
X = \frac{N-\Delta\beta m p-\Delta\delta\sec\theta}{\Delta\alpha\mu}
$$

^c55034

### Intensities of solar radiation
These are measured by Dobson Spectrophotometer.

$N=L_0-L=\log{I_0/I_0^\prime}-\log{I/I^\prime}$ 

$L_0$ is "extra-terrestrial constant", a hypothetical measurement by Dobson Spectrophotometer outside of atmosphere
$I_0$ (short wavelength) and $I_0^\prime$ (long wavelength) - out-of-atmosphere intensities. 
 $I$ and $I^\prime$ - ground-level intensities. 
 
These are obtained from measurement of an R-dial in section [[2 R-Dial to N-Values]]

### Zenith angle
* $\theta$ - solar zenith angle (angular zenith distance of the sun); with $0^\degree$ as sun directly overhead, 90 at the horizon

### Deltas
* $\Delta\alpha$ - Absorption coefficients delta between wavelengths (tabulated [doc 2]).
* $\Delta\beta$ - Air Rayleigh scattering coefficient delta between wavelengths (tabulated [doc 2]).
* $\Delta\delta$ - Scattering coefficient of aerosol particles delta (zeroed out for $X_{12}$ below)

### Ratios
* $p$ - ratio of observed station pressure and mean sea level pressure
	1. For Halley station, which is at 20-30m above sea level, $p\approx1$
	2. Can be found in the meteorological data from a station, using SYNOP [doc 2]

#### Actual and vertical solar radiation paths ratios:
* $m$ - path through the atmosphere ratio (accounts for refraction and the earth's curvature)

* $\mu$ - path through the ozone layer ratio. 

	$$\mu=\frac{R+h}{\sqrt{(R+h)^2-(R+r)^2\sin^2{\gamma}}}$$
	$R$ - Earch radius
	$r$ - height of station above sea level
	$h$ - height of ozon layer: 
		1. Either measured or table-interpolated [doc 2]
		2. Has seasonal dependence
		3. It may be possible to use Dobson data to derive the profile of the ozone layer - not attempted yet at BAS
	$\gamma$ - solar zenith angle (SZA)
	
## Two wavelength pairs
When difference in $X$ is measured between the two wavelength pairs as $X_{12}$, the expression is simplified because some of the deltas are zeroed out ($\Delta\delta_{12}\approx0$):

$$X_{12}=\frac{N_{12}}{\Delta\alpha_{12}\mu}-\frac{\Delta\beta_{12}p}{\Delta\alpha_{12}\mu}$$



