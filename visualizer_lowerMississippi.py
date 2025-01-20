import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

results_path = "lowerMississippi/output/"
actual_path = "lowerMississippi/donaldsville/"
baton_path = "lowerMississippi/baton rouge/"

res_time = np.loadtxt(results_path + "time.txt")
res_dsQ = np.loadtxt(results_path + "downstreamQ1000.txt")


columns = [
    "agency_cd", "site_no", "datetime", "gage_height_max", "max_cd",
    "gage_height_min", "min_cd", "gage_height_mean", "mean_cd", "discharge", "discharge_cd"
]
data = pd.read_csv(actual_path + "usgs.txt", sep='\t', comment='#', names=columns, header=1, parse_dates=["datetime"])
data = data[["datetime", "gage_height_mean", "discharge"]]
data.dropna(inplace=True)

feet_to_meters = 0.3048  # 1 foot = 0.3048 meters
cfs_to_cms = 0.0283168

data["gage_height_mean"] *= feet_to_meters
data["discharge"] *= cfs_to_cms

res_date = np.array([data["datetime"][0] + timedelta(seconds=sec) for sec in res_time])

data_baton = pd.read_csv(baton_path + "usgs.txt", sep='\t', comment='#', names=columns, header=1, parse_dates=["datetime"])
data_baton = data_baton[["datetime", "gage_height_mean", "discharge"]]
data_baton.dropna(inplace=True)
data_baton["gage_height_mean"] *= feet_to_meters
data_baton["discharge"] *= cfs_to_cms

plt.plot(data["datetime"], data["discharge"], label="USGS data (Donaldsville)", color = "k")
plt.plot(data_baton["datetime"], data_baton["discharge"], "o", label="USGS data at Baton Rouge", alpha=0.1, color="k")
plt.plot(res_date, res_dsQ, "*", label="Model results at Donaldsville", color="k")
plt.xlabel("Date")
plt.ylabel("Discharge ($m^3/s$)")
plt.legend()
plt.title("Discharge at Donaldsville")
plt.show()

"""add water level graphs also"""