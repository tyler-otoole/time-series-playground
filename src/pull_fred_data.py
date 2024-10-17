import pyfredapi as pf

gdp_info = pf.get_series_info(series_id = "GDP")

print(gdp_info)