import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

dtdxinfo = "dx500dt500"
dxinfo = "500"

results_path = "lowerMississippi/output" + dtdxinfo + "/"
actual_path = "lowerMississippi/donaldsville/"
baton_path = "lowerMississippi/baton rouge/"

res_time = np.loadtxt(results_path + "time.txt")
res_dsQ = np.loadtxt(results_path + "downstreamQ" + dxinfo + ".txt")
res_dsH = np.loadtxt(results_path + "downstreamH" + dxinfo + ".txt")

feet_to_meters = 0.3048  # 1 foot = 0.3048 meters
cfs_to_mcs = 0.0283168

columns = [
    "agency_cd", "site_no", "datetime", "gage_height_max", "max_cd",
    "gage_height_min", "min_cd", "gage_height_mean", "mean_cd", "discharge", "discharge_cd"
]
data_baton = pd.read_csv(baton_path + "usgs.txt", sep='\t', comment='#', names=columns, header=1, parse_dates=["datetime"])
data_baton = data_baton[["datetime", "gage_height_mean", "discharge"]]

data_dv = pd.read_csv(actual_path + "gageH.txt", skiprows=0)
data_dv["datetime"] = data_baton["datetime"]
data_dv["gageH"] = pd.to_numeric(data_dv["gageH"], errors="coerce")
data_dv["gageH"] *= feet_to_meters
data_dv.dropna()

res_date = np.array([data_baton["datetime"][0] + timedelta(seconds=sec) for sec in res_time])

plt.plot(res_date, res_dsH, label="Model results")
plt.plot(data_dv["datetime"], data_dv["gageH"], label="USACE data")
plt.xlabel("Date")
plt.ylabel("Water Level ($m$)")
plt.legend()
plt.title("Water level at Donaldsville (downstream)")
plt.show()


