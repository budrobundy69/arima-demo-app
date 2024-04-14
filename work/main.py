import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Daten generieren
data = {
  'Datum': pd.date_range(start='1/1/2020', periods=100),
  'Temperatur': pd.Series(20 + np.random.randn(100).cumsum())
}

df = pd.DataFrame(data).set_index('Datum')

# ARIMA-Modell anpassen
model = ARIMA(df['Temperatur'], order=(1, 1, 1))
model_fit = model.fit()

# Vorhersagen und Plot
df['forecast'] = model_fit.predict(start=90, end=99, dynamic=True)
df[['Temperatur', 'forecast']].plot()
plt.show()
