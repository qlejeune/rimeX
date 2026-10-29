# SPDX-License-Identifier: AGPL-3.0-or-later

from rimeX.preproc.quantilemaps import make_timesensitive_quantilemap_prediction
import pandas as pd
import numpy as np 
import xarray as xr
from rimeX.datasets.manager import register_dataset
from sklearn.linear_model import LinearRegression
import warnings

_require_unwpp_lifeexpectancy = register_dataset(
    "unwpp/WPP2024_MORT_F05_1_LIFE_EXPECTANCY_BY_AGE_BOTH_SEXES.xlsx",
    url="https://population.un.org/wpp/assets/Excel%20Files/1_Indicator%20(Standard)/EXCEL_FILES/4_Mortality/WPP2024_MORT_F05_1_LIFE_EXPECTANCY_BY_AGE_BOTH_SEXES.xlsx",
)

_require_unwpp_lifetable_hist = register_dataset(
    "unwpp/WPP2024_MORT_F06_1_SINGLE_AGE_LIFE_TABLE_ESTIMATES_BOTH_SEXES.xlsx",
    url="https://population.un.org/wpp/assets/Excel%20Files/1_Indicator%20(Standard)/EXCEL_FILES/4_Mortality/WPP2024_MORT_F06_1_SINGLE_AGE_LIFE_TABLE_ESTIMATES_BOTH_SEXES.xlsx",
)

_require_unwpp_lifetable_proj = register_dataset(
    "unwpp/WPP2024_MORT_F06_1_SINGLE_AGE_LIFE_TABLE_PROJECTIONS_BOTH_SEXES.xlsx",
    url="https://population.un.org/wpp/assets/Excel%20Files/1_Indicator%20(Standard)/EXCEL_FILES/4_Mortality/WPP2024_MORT_F06_1_SINGLE_AGE_LIFE_TABLE_PROJECTIONS_BOTH_SEXES.xlsx",
)

def download_unwpp_lifeexpectancy():
    return _require_unwpp_lifeexpectancy()

def download_unwpp_lifetable_hist():
    return _require_unwpp_lifetable_hist()

def download_unwpp_lifetable_proj():
    return _require_unwpp_lifetable_proj()

def prepare_unwpp_lifetable(
        filepath_lifetable_hist=None,
        filepath_lifetable_proj=None
):
    """
    Load variables from the UNWPP2024 life tables that are required to calculate cohort life expectancy.
    Data are provided separately for historical (1950-2023) and projection (2024-2100) periods.
    ...
    """
    
    if filepath_lifetable_hist is None:
        filepath_lifetable_hist = download_unwpp_lifetable_hist()
    if filepath_lifetable_proj is None:
        filepath_lifetable_proj = download_unwpp_lifetable_proj()

    def read_sheets_unwppp_lifetables(filepath_lifetable):
        parts = []
        with pd.ExcelFile(filepath_lifetable) as xl:
            for name in xl.sheet_names:
                raw = xl.parse(name, skiprows=16)
                if 'Type' not in raw.columns:
                    print(f"skipped sheet: {name}")
                    continue
                df = raw[raw['Type'] == 'Country/Area'].rename(
                    columns={'Region, subregion, country or area *': 'Country'}
                )
                df = df[['Country', 'ISO3 Alpha-code', 'Year', 'Age (x)',
                         'Central death rate m(x,n)', 'Average number of years lived a(x,n)']]
                parts.append(df)
        return pd.concat(parts, ignore_index=True)

    df_allyears = pd.concat([read_sheets_unwppp_lifetables(filepath_lifetable_hist), read_sheets_unwppp_lifetables(filepath_lifetable_proj)], ignore_index=True)
    
    # Tests: Check data integrity
    iso = 'ISO3 Alpha-code'
    name = 'Country'

    ## Missing or empty ISO3 codes
    n_bad = (df_allyears[iso].isna() | (df_allyears[iso].astype(str).str.strip() == '')).sum()
    assert n_bad == 0, f"{n_bad} lignes sans code ISO3 valide"

    ## A same ISO3 code is associated to only one country name
    n_names = df_allyears.groupby(iso)[name].nunique()
    assert (n_names == 1).all(), f"Codes with several country names :\n{n_names[n_names > 1]}"

    ## A same country name is associated to only one ISO3 code
    n_codes = df_allyears.groupby(name)[iso].nunique()
    assert (n_codes == 1).all(), f"Country names with several codes :\n{n_codes[n_codes > 1]}"

    return df_allyears


def _make_tables(df_country):
    """Return one Year x Age table per variable.
    pivot raises an error if (Year, Age) is duplicated."""
    return {v: df_country.pivot(index='Year', columns='Age (x)', values=v).sort_index()
            for v in ('Central death rate m(x,n)', 'Average number of years lived a(x,n)'
    )}


def _diagonal(tbl, birth_year, shift=0):
    """Values along the cohort diagonal: age a is read in year b + a + shift.
    NaN where the year doesn't exist in the table."""
    ages = tbl.columns.to_numpy()
    r = tbl.index.get_indexer(birth_year + ages + shift)   # -1 where the year is missing
    arr = tbl.to_numpy(dtype=float)
    return np.where(r >= 0, arr[r, np.arange(len(ages))], np.nan)


def _fill_life_table(ct, radix=100_000):
    """Calculates all missing columns from the life table.
    The last value calculated is the life expectancy at all ages, including at birth."""
    m = ct['death rate m(x,n)'].to_numpy()
    a = ct['years lived a(x,n)'].to_numpy()
    n = 1

    q = n * m / (1 + (n - a) * m)            # probability of dying in the interval
    q[-1] = 1.0                              # open last interval: everyone dies

    l = np.empty(len(ct))
    l[0] = radix
    l[1:] = radix * np.cumprod(1 - q[:-1])   # survivors at exact age x

    d = l * q

    L = (l - d) * n + d * a                  # person-years lived in the interval
    L[-1] = l[-1] / m[-1]                    # open last interval

    T = np.cumsum(L[::-1])[::-1]             # total person-years lived above age x

    e = T / l                                # life expectancy at exact age x

    ct['probability of dying q(x,n)'] = q
    ct['number of survivors l(x)'] = l
    ct['number of deaths d(x,n)'] = d
    ct['person-years lived L(x,n)'] = L
    ct['years left to live T(x)'] = T
    ct['life expectancy e(x)'] = e
    return ct


def _cohort_table(tables, birth_year):
    ages = tables['Central death rate m(x,n)'].columns.to_numpy()
    mx_n  = _diagonal(tables['Central death rate m(x,n)'], birth_year, shift=0)   # age a in year b + a
    mx_n1 = _diagonal(tables['Central death rate m(x,n)'], birth_year, shift=1)   # age a in year b + a + 1
    ax_n  = _diagonal(tables['Average number of years lived a(x,n)'], birth_year, shift=0)   # age a in year b + a
    ax_n1 = _diagonal(tables['Average number of years lived a(x,n)'], birth_year, shift=1)   # age a in year b + a + 1
    return pd.DataFrame({
        'Age': ages,
        'Year at age': birth_year + ages,
        'death rate m(x,n)': (mx_n + mx_n1) / 2,
        'years lived a(x,n)': (ax_n + ax_n1) / 2,
    })


def _extend_table(tables, last_birth_year):
    """
    Extend tables with mortality rate or time spent with missing values. 
    First identify years with  missing values, then fit a trend over the last years with data to extrapolate and get missing data.
    """
    for v in ('Central death rate m(x,n)', 'Average number of years lived a(x,n)'):
        
        # Identify how many years are missing in the diagonal for the last birth year requested (with shift=1)
        # This gives us the ages that have missing values. 
        Nyr_missing = np.count_nonzero(np.isnan(_diagonal(tables[v], last_birth_year, shift=1)))
        ages = tables[v].columns.to_numpy()
        cols = ages[-Nyr_missing:]
        last_year_nonmissing = tables[v][cols[0]].last_valid_index()

        # Fit a trend over the last 20 years with data for all columns (age groups) with missing data 
        fit_years = np.arange(last_year_nonmissing - 19, last_year_nonmissing + 1)
        Y = tables[v].loc[fit_years, cols]
        assert Y.notna().all().all(), ("NaN in the 20-year fitting window")
        model = LinearRegression().fit(fit_years.reshape(-1, 1),Y.to_numpy())

        # Extrapolate the trends to fill in the missing years
        new_years = np.arange(last_year_nonmissing + 1, last_year_nonmissing + Nyr_missing + 1)
        pred = model.predict(new_years.reshape(-1, 1))

        # Issue warning if the mortality rate becomes negative
        # Only cells actually used by the cohort diagonals: age a is needed up to year last_birth_year + 1 + a
        needed = new_years[:, None] <= (last_birth_year + 1 + cols)[None, :]
        neg = (pred < 0) & needed

        if neg.any():
            bad_ages = cols[neg.any(axis=0)]
            warnings.warn(
                f"{label}: negative extrapolated '{v}' for ages {bad_ages.tolist()} "
                f"(min = {pred[neg].min():.3g})",
                stacklevel=2,
            )

        # Extend the index, then fill
        tables[v] = tables[v].reindex(tables[v].index.union(new_years))
        tables[v].loc[new_years, cols] = pred

    return tables


def calc_life_expectancy(df_unwpp_allyears, first_birth_year, last_birth_year):
    """
    Calculate cohort life expectancy at birth and at age 5 for each country and year of birth.
    ...
    """
    rows = []
    for iso_code, g in df_unwpp_allyears.groupby('ISO3 Alpha-code'):
        country = g['Country'].iloc[0]
        tables = _make_tables(g)

        # Check that the tables contain all years for the requested birth years
        # check if we have NaN values in the diagonal for the last birth year requested (with shift=1)
        diag = _diagonal(tables['Central death rate m(x,n)'], last_birth_year, shift=1)
        if np.isnan(diag).any():
            print(f'Extrapolating data for {country}')
            tables = _extend_table(tables, last_birth_year)

        for birth_year in range(first_birth_year, last_birth_year + 1):
            ct = _fill_life_table(_cohort_table(tables, birth_year), radix=100_000)
            e = ct.set_index('Age')['life expectancy e(x)']

            if e.isna().any():
                bad_ages = e.index[e.isna()].tolist()
                warnings.warn(
                    f"NaN in life table for {country} ({iso_code}), "
                    f"birth year {birth_year}: ages {bad_ages[:5]}"
                    f"{'...' if len(bad_ages) > 5 else ''}",
                    stacklevel=2,
                )

            rows.append({'ISO3': iso_code,
                         'Country': country,
                         'Birth Year': birth_year,
                         'e0': e.loc[0],
                         'e5': e.loc[5]})

    df_e_long = pd.DataFrame(rows)

    df_life_expectancy_at_birth = df_e_long.pivot(index='Birth Year',
                             columns=['Country', 'ISO3'],
                             values='e0')
    df_life_expectancy_at_age5 = df_e_long.pivot(index='Birth Year',
                             columns=['Country', 'ISO3'],
                             values='e5')
    return df_life_expectancy_at_birth, df_life_expectancy_at_age5


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

    countries = [country] if isinstance(country, str) else list(country)
    birth_years = np.arange(start_birthyear, end_birthyear + 1)

    # --- restrict the dataset to the requested countries
    has_country = 'country' in ds_ts_quantiles.coords
    if has_country:
        if 'country' not in ds_ts_quantiles.dims:       # scalar coord (e.g. after .sel)
            ds_ts_quantiles = ds_ts_quantiles.expand_dims('country')
        ds_ts_quantiles = ds_ts_quantiles.sel(country=countries)   # KeyError if absent
    elif len(countries) > 1:
        raise ValueError("Several countries requested but the data has no 'country' coordinate.")

    # --- life expectancy, one column per requested country
    le_full = df_life_expectancy[countries].astype(float)
    le = le_full.loc[birth_years]

    # --- Stop if lifetime extends beyond available quantile data
    max_death_yr = int(np.floor((np.expand_dims(birth_years, 1) + le.values).max()))
    current_max_year = int(ds_ts_quantiles["year"].max().values)

    if max_death_yr > current_max_year:
        valid = ((le_full.index.values[:, None] + le_full.values) <= current_max_year).all(axis=1)
        max_possible_birthyear = le_full.index[valid].max()
        raise ValueError(
            f"max death year ({max_death_yr}) exceeds max year in "
            f"RIME-X quantiles ({current_max_year}); "
            f"maximum possible birth year is {max_possible_birthyear}"
        )

    # --- Identify expected death year and fraction of last year partially lived
    life_exp = xr.DataArray(
        le.values,
        coords={"birth_year": birth_years, "country": countries},
        dims=("birth_year", "country")
    )
    if not has_country:                                  # data without country dimension
        life_exp = life_exp.isel(country=0, drop=True)

    death_year = life_exp["birth_year"] + np.floor(life_exp)
    fraction_lastyr = life_exp - np.floor(life_exp)

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
