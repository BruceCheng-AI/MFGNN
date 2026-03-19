# Dataset Dictionary

## Processed Tables

| File | Role | Key columns observed |
| --- | --- | --- |
| `data/processed/traffic.csv` | street-level target table | `time`, `street_id`, `street_name`, `district_name`, `district_id`, `EXPONENT`, `SPEED` |
| `data/processed/weather_street.csv` | street-level weather features | `DDATETIME`, `R1h`, `W1h`, `T1h`, `V1h`, `street_name`, `district_name`, `district_id`, `street_id` |
| `data/processed/events.csv` | event intensity aligned to streets | `DDATETIME`, `Street`, `Overall Rating (1-10)`, `street_name`, `district_name`, `district_id`, `street_id` |
| `data/processed/warning_weather.csv` | warning-weather labels by street | `DDATETIME`, `Type`, `Color`, `WARNING_Type`, `WARNING_COLOR`, `WARNING_SEVERITY`, `street_name`, `district_name`, `district_id`, `street_id` |
| `data/processed/holidays.csv` | holiday labels by street and time | `time`, `street_id`, `label` |
| `data/processed/forecast.csv` | district forecast source table | `AREANAME`, `FORECASTTIME`, `WEATHERSTATUS`, `DDATETIME` |
| `data/processed/forecast_filled.csv` | cleaned and street-aligned forecast table | `FORECASTTIME`, `WEATHERSTATUS`, `DDATETIME`, `FORECAST_HOUR`, `WEATHER_SEVERITY`, `street_name`, `district_name`, `district_id`, `street_id` |
| `data/processed/shenzhen_street_adjacency_matrix.csv` | street adjacency matrix | wide matrix indexed by street names |
| `data/processed/street_district_mapping.csv` | street-to-district lookup | `street_name`, `district_name`, `district_id`, `street_id` |

## LibCity-Style Dataset Files

| File | Role | Key columns observed |
| --- | --- | --- |
| `StreetSZ/StreetSZ.dyna` | target dynamics in LibCity format | `dyna_id`, `type`, `time`, `entity_id`, `traffic_speed`, `TPI` |
| `StreetSZ/StreetSZ.ext` | exogenous features in LibCity format | `ext_id`, `time`, `geo_id`, `R1h`, `W1h`, `T1h`, `V1h`, `alert_level`, `holiday_status`, `event_rating` |
| `StreetSZ/StreetSZ.fut` | future indicators in LibCity format | `time`, `geo_id`, `weather_forecast`, `holiday_status`, `event_rating`, `fut_id` |
| `StreetSZ/StreetSZ.his` | historical summary features | `time`, `geo_id`, `closeness_TPI`, `closeness_speed`, `period_TPI`, `period_speed`, `trend_TPI`, `trend_speed`, `his_id` |
| `StreetSZ/StreetSZ.geo` | node geometry and static attributes | `geo_id`, `type`, `coordinates`, `DISTRICT_ID`, `Area`, `RoadDensity`, `BuildingArea`, `CarStation`, `CarPark`, `Subway`, `POI` |
| `StreetSZ/StreetSZ.rel` | graph relations | `rel_id`, `type`, `origin_id`, `destination_id`, `link_weight` |
| `StreetSZ/config.json` | dataset schema config | data columns, ext columns, weight settings, time interval |

## Raw Source Files

| File or folder | Role | Notes |
| --- | --- | --- |
| `data/raw/traffic_flow.csv` | raw traffic source table | includes traffic and contextual attributes |
| `data/raw/weather.csv` | raw weather table | example columns include `DDATETIME`, `DISTRICT_NAME`, `R1h`, `W1h`, `T1h`, `V1h` |
| `data/raw/Event_En.csv` | raw event table | includes event name, date range, location, street, scale, impact, and rating |
| `data/raw/2018_holidays.csv` | holiday source table | small holiday label source |
| `data/raw/Forecast/` | raw forecast snapshots | many `Forecast*.csv` files |
| `data/raw/Traffic/` | raw traffic snapshots | many time-stamped CSV files |
| `data/raw/Street/` | raw spatial street files | includes shapefile components and `roadnet_sz.csv` |
| `data/raw/Street_Attr.csv` | raw street attribute table | static street attributes |
| `data/raw/sz_geo.csv` | raw spatial metadata | auxiliary geo table |
| `data/raw/History_inf2.csv` | raw historical summary source | auxiliary history table |

## Suggested Public Naming Convention

- keep the processed package as the main public dataset
- release the raw source package only if redistribution is legally allowed
- version both packages independently if they are uploaded separately
