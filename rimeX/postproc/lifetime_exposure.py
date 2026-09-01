# SPDX-License-Identifier: AGPL-3.0-or-later

from rimeX.preproc.quantilemaps import make_timesensitive_quantilemap_prediction
import pandas as pd
import numpy as np 
import xarray as xr
from rimeX.datasets.manager import register_dataset

_require_unwpp_lifeexpectancy = register_dataset(
    "unwpp/WPP2024_MORT_F05_1_LIFE_EXPECTANCY_BY_AGE_BOTH_SEXES.xlsx",
    url="https://population.un.org/wpp/assets/Excel%20Files/1_Indicator%20(Standard)/EXCEL_FILES/4_Mortality/WPP2024_MORT_F05_1_LIFE_EXPECTANCY_BY_AGE_BOTH_SEXES.xlsx",
)

def download_unwpp_lifeexpectancy():
    return _require_unwpp_lifeexpectancy()


def load_unwpp_lifeexpectancy(
        filepath_lifeexpectancy=None,
        start_birthyear=1950,
        end_birthyear=2025
):
    """
    Load UNWPP2024 data on e(x) = Life Expectancy at Exact Age x (ex) - Both Sexes.
    ...
    """
    if filepath_lifeexpectancy is None:
        filepath_lifeexpectancy = download_unwpp_lifeexpectancy()

    df_list = []
    for sheet in [0, 1]:
        df_unwpp_raw = pd.read_excel(filepath_lifeexpectancy,
                sheet_name=sheet,
                skiprows=16)

        df_unwpp = df_unwpp_raw[df_unwpp_raw['Type'] == 'Country/Area'].rename(
                columns={'Region, subregion, country or area *': 'Country'})

        cols = df_unwpp.columns
        idxs = [i for i, col in enumerate(cols) if col in ('Country', 'ISO3 Alpha-code', 'Year', 5)]
        df_unwpp = df_unwpp.iloc[:, idxs].pivot(
            index='Year',
            columns=['Country', 'ISO3 Alpha-code'],
            values=5)

        df_unwpp.index = df_unwpp.index.astype(int)
        df_list.append(df_unwpp)

    df_unwpp = pd.concat(df_list, axis=0).loc[start_birthyear:int(end_birthyear + 5)]
    return df_unwpp

def get_life_expectancies(df_unwpp,
                         start_birthyear=1950,
                         end_birthyear=2025):
    
    """
    Takes UNWPP life expectancy data expressed as years left to live at age of 5, 
    subtracts 5 from Year to get it at birth year but ignoring infant mortality, 
    adds 5 to account for the 5 years of life already lived, adds 6 to account for increase 
    in life expectancy through the life of an individual (i.e. move from "period" life expectancy to 
    "cohort" life expectancy, see Goldstein & Wachter (2006) "Relationships between period and cohort 
    life expectancy: Gaps and lags")

    Thus get life expectancy in each year for each country at birth 
    expressed in "cohort" way, neglecting infant mortality.

    Adter end of data, extends by filling with constant value 

    Inputs


    Returns 

    """
    
    df_life_expectancy_5 = df_unwpp.copy()
    df_life_expectancy_5.index = df_life_expectancy_5.index-5 # year of birth: 2023 (age 5) becomes 2018 (age 0)
    df_life_expectancy_5 = df_life_expectancy_5 + 5 + 6 

    if df_life_expectancy_5.index[-1] < end_birthyear :
        # extend for last years
        df_life_expectancy_5_extend = df_life_expectancy_5.reindex(
                    np.arange(start_birthyear,end_birthyear+1)).astype( 
                    'float').interpolate() # extrapolation: fills last years constant 
    
        return df_life_expectancy_5_extend
    else:
        return df_life_expectancy_5.loc[start_birthyear:end_birthyear ]

def calc_lifetime_exposure_rimex(ds_ts_quantiles, df_life_expectancy,
                                 start_birthyear, end_birthyear, country):

    # ds_ts_quantiles should have:
    # dims: quantile, year
    # data_vars: tuples like ('HI-caution-QDM-v1', 'NGFS CurPol') or just ('HI-caution-QDM-v1')

    # Determine maximum death year and maximum year currently in the dataset
    birth_years_check = np.arange(start_birthyear, end_birthyear + 1)

    life_exp_check = df_life_expectancy.loc[birth_years_check, country]
    death_years = birth_years_check + life_exp_check.values

    max_death_yr = int(np.floor(death_years.max()))
    current_max_year = int(ds_ts_quantiles["year"].max().values)

    # Stop if lifetime extends beyond available quantile data
    if max_death_yr > current_max_year:

        valid = (
            df_life_expectancy.index
            + df_life_expectancy[country]
            <= current_max_year
        )

        max_possible_birthyear = df_life_expectancy.index[valid].max()

        raise ValueError(
            f"max death year ({max_death_yr}) exceeds max year in "
            f"RIME-X quantiles ({current_max_year}); "
            f"maximum possible birth year is {max_possible_birthyear}"
        )

    birth_years = np.arange(start_birthyear, end_birthyear + 1)

    life_exp = xr.DataArray(
        df_life_expectancy.loc[birth_years, country].values,
        coords={"birth_year": birth_years},
        dims="birth_year"
    )

    death_year = xr.DataArray(
        birth_years + np.floor(life_exp.values),
        coords={"birth_year": birth_years},
        dims="birth_year"
    )

    fraction_lastyr = xr.DataArray(
        life_exp.values - np.floor(life_exp.values),
        coords={"birth_year": birth_years},
        dims="birth_year"
    )

    # add birth_year dimension
    ds_expanded = ds_ts_quantiles.expand_dims(birth_year=birth_years)

    year = ds_expanded["year"]
    birth = ds_expanded["birth_year"]

    # full years: birth year through year before death year
    full_mask = (year >= birth) & (year <= (death_year - 1))
    exposure_fullyrs = ds_expanded.where(full_mask).sum(dim="year")

    # fractional final year
    partial_mask = year == death_year
    exposure_lastyr = (
        ds_expanded.where(partial_mask).sum(dim="year") * fraction_lastyr
    )

    ds_lifetime_exp = exposure_fullyrs + exposure_lastyr

    return ds_lifetime_exp
