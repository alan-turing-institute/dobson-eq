# `Z6_Dataframe_constants.csv`
'Year'
'Lo (A)'
'Lo (C) '
'Lo (D)'
'Lat'
'Long'
'ID': str, '031E' or '073E'



# `Z6_Dataframe_zenith.csv`
'dt'
'type'
'Dobson'
'Lat'
'Lon'
'zenith_angle',
'100hPa T'
'Lo(A)'
'Lo(C)'
'Lo(D)'
'A1', 'A2', 'A3',
'C1', 'C2','C3', 
'D1', 'D2', 
'L1(A)', 'L2(A)', 'L3(A)', 
'L1(C) ', 'L2(C)', 'L3(C)',
'L1(D)', 'L2(D)', 
'A', 'C', 'D', 
'L(A)', 'L(C)', 'L(D)',

# `Z6_Dataframe_direct.csv`:

## Timestamp
* `dt` - timestamp
* `month`
* `year`

## Position
* `Lat` float
* `Lon` float

## Sun angle
* `cosSZA` float
* `zenith_angle` float
* `secSZA` float

## mu
* `mu` float
* `invmu` float

## Intensities of wavelengths
* ? `Lo(A)` float
* ? `Lo(C)` float 
* ? `Lo(D)` float

* ? `L(A)` float
* ? `L(C)` float
* ? `L(D)` float

 * `L1(A)`, `L2(A)`, `L3(A)` float
 * `L1(C)`, `L2(C)`, `L3(C)` float
 * `L1(D)`, `L2(D)` float

9. `A1`, `A2`, `A3` float
10. `C1`, `C2`, `C3` float
11. `D1`, `D2` int
15. `A`, `C`, `D` float


## Ozone
* `OZAD` float
* `OZCD` float
* `D073E` bool (a.s. False)
* `D031E` bool (a.s. True)

## Pressure
? `100hPa T` float

## Unknown
3. ? `type1` (integer, a.s. 2)
4. ? `type2` (integer, a.s. 20 or 90 )
5. ? `Dobson` string, a.s. '031E'




## Postprocess
* `n1C_m_n2C` float
* `n1D_m_n2D` float
* `n1A_m_n2A`
* `nC_m_nD` 
* `nA_m_nD`
* `nC_o_nD`
* `nA_o_nD`
* `nA_m_nD_o_mu`
* `nC_m_nD_o_mu`



