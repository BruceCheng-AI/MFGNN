# Dataset Card

## Dataset Name

Suggested public name:

`Shenzhen Street-Level Traffic, Weather, Holiday, and Event Dataset for Traffic Congestion Prediction`

## Summary

This dataset supports street-level traffic congestion prediction in Shenzhen. The target variables are `traffic_speed` and `TPI`. The inputs combine historical traffic measurements, weather features, warning-weather signals, holiday labels, event ratings, spatial relations, node attributes, and future exogenous indicators.

## Recommended Public Packages

### Core Reproducibility Package

- `data/processed/traffic.csv`
- `data/processed/weather_street.csv`
- `data/processed/events.csv`
- `data/processed/warning_weather.csv`
- `data/processed/holidays.csv`
- `data/processed/forecast.csv`
- `data/processed/forecast_filled.csv`
- `data/processed/shenzhen_street_adjacency_matrix.csv`
- `data/processed/street_district_mapping.csv`
- `StreetSZ/`

### Optional Raw Supplementary Package

- `data/raw/traffic_flow.csv`
- `data/raw/weather.csv`
- `data/raw/Event_En.csv`
- `data/raw/2018_holidays.csv`
- `data/raw/Forecast/`
- `data/raw/Traffic/`
- `data/raw/Street/`
- `data/raw/Street_Attr.csv`
- `data/raw/sz_geo.csv`
- `data/raw/History_inf2.csv`

## Spatial Unit

- street-level entities represented in `StreetSZ.geo`
- district metadata linked through `district_name`, `district_id`, and `street_district_mapping.csv`

## Temporal Resolution

- the LibCity dataset config declares `time_intervals = 10800`, which corresponds to 3-hour intervals

## Targets

- `traffic_speed`
- `TPI`

## Main Exogenous Signals

- `R1h`, `W1h`, `T1h`, `V1h`
- `alert_level`
- `holiday_status`
- `event_rating`
- forecast-derived indicators in `forecast.csv` and `forecast_filled.csv`

## Files Used By The Public Modeling Pipeline

- `StreetSZ.dyna`: target dynamics
- `StreetSZ.ext`: exogenous features
- `StreetSZ.fut`: future indicators
- `StreetSZ.his`: historical summary features
- `StreetSZ.geo`: node attributes
- `StreetSZ.rel`: graph relations
- `StreetSZ/config.json`: dataset schema

## Suggested Metadata To Publish Alongside The Dataset

- title
- version
- release date
- authors and affiliations
- contact email
- data license
- citation text
- checksum list
- preprocessing summary

## Ethics And Redistribution Notes

- Recheck redistribution rights for every upstream source before release.
- Confirm that the released files do not expose restricted data or private annotations.
- If any raw source cannot be redistributed, publish only the processed package and document the restriction clearly.
