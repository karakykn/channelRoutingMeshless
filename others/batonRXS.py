import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

file_path = "../LowerMississippi/baton rouge/fieldmeasurements.txt"
data = pd.read_csv(file_path, sep='\t')
new_df = data.loc[200:, ["gage_height_va", "chan_width", "chan_area"]]
new_df = new_df.astype(float)

new_df = new_df.dropna()
ft_mt = 0.3048
ft2_mt2 = ft_mt**2

new_df["gage_height_va"] = new_df["gage_height_va"] * ft_mt
new_df["chan_width"] = new_df["chan_width"] * ft_mt
new_df["chan_area"] = new_df["chan_area"] * ft2_mt2

# new_df["area_wh"] = new_df["chan_area"] / new_df["chan_width"] / new_df["gage_height_va"]

X = new_df[["gage_height_va"]]  # Independent variable (feature)
y = new_df["chan_width"]    # Dependent variable (target)

df_pred = pd.DataFrame({
    "gage_height_va": np.linspace(0,3*np.max(X),400)
})
xx = df_pred[["gage_height_va"]]

# Initialize and fit the model
model = LinearRegression()
model.fit(X, y)

# Print the results
# print(f"Coefficient width: {model.coef_}")
# print(f"Intercept: {model.intercept_}")
# width_sq = model.coef_

y_pred = model.predict(xx)

df_pred["chan_width_pred"] = y_pred

# plt.plot(new_df["gage_height_va"], new_df["chan_width"], "*")
# plt.plot(df_pred["gage_height_va"], df_pred["chan_width_pred"])
# plt.show()

areaP_sq = new_df["chan_area"].apply(np.sqrt)
model.fit(X,areaP_sq)
area_pred_sq = model.predict(xx)
area_pred = area_pred_sq ** 2
df_pred["chan_area_pred"] = area_pred
df_pred["hydra_radius"] = df_pred["chan_area_pred"] / df_pred["chan_width_pred"]
"""above assumes wetted perimeter is equal to the channel width"""

df_pred.to_csv('../lowerMississippi/riverProfile.txt', sep=',', index=False)