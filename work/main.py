import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# ---- MAGIC -----
# Daten generieren
data = {
  'Datum': pd.date_range(start='1/1/2020', periods=100),
  'Temperatur': pd.Series(20 + np.random.randn(100).cumsum())
}

df = pd.DataFrame(data).set_index('Datum')
df
print('Shape:', df.shape)
df.head()

# ARIMA-Modell anpassen
model = ARIMA(df['Temperatur'], order=(1, 1, 1))
model_fit = model.fit()

# Vorhersagen und Plot
df['forecast'] = model_fit.predict(start=90, end=99, dynamic=True)
df[['Temperatur', 'forecast']].plot()
plt.show()
