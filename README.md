# Integrating Alternative Sources Ensures Water Supply, but Might Not Fully Meet Future Agricultural Needs in Northeast Brazil

This repository contains the data and code used in the analysis for the paper titled:  
**"Integrating Alternative Sources Ensures Water Supply, but Might Not Fully Meet Future Agricultural Needs in Northeast Brazil."**

## Repository Contents

- **`water-supply-optimization.jl`**  
  Julia script implementing the water supply optimization model used in the study.

- **`cav_reservoirs.csv`**  
  Reservoir storage capacity data.

- **`evaporation_reservoirs.csv`**  
  Monthly evaporation data for key reservoirs.

- **`inflow_series.xlsx`**  
  Historical inflow time series data for the modeled system.

- **`volume_max_reservoirs.csv`**  
  Maximum volume constraints for each reservoir.

## Usage

This project uses [Julia](https://julialang.org/). Make sure to have Julia installed, and run the script using the Julia REPL or your preferred IDE.

```bash
julia water-supply-optimization.jl