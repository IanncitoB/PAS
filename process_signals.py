import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet


def plot_series(series, title=None, xlabel='Date', ylabel='Demanda (MW)'):
    """
    Plot the entire time series.
    series: pandas Series with DateTime index.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(series)
    if title:
        plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()


def moving_average(series, window=30):
    """
    Compute moving average of the series.
    window: window size (int).
    Returns a pandas Series.
    """
    return series.rolling(window=window, center=False).mean()


def plot_moving_average(series, ma, window=30, title='Moving Average', xlabel='Date', ylabel='Demanda (MW)'):
    """
    Plot moving average on top of the series.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(series, label='Original')
    plt.plot(ma, label=f'MA{window}')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()


def seasonality_tests(series):
    """
    Perform ADF and KPSS tests for stationarity.
    Returns dict with test statistics and p-values.
    """
    adf_res = adfuller(series.dropna())
    kpss_res = kpss(series.dropna(), nlags='auto')
    return {
        'adf_stat': adf_res[0],
        'adf_pvalue': adf_res[1],
        'adf_crit': adf_res[4],
        'kpss_stat': kpss_res[0],
        'kpss_pvalue': kpss_res[1],
        'kpss_crit': kpss_res[3]
    }

def print_stationarity_tests(results, name):
    if results['adf_pvalue'] > 0.05:
        resumen = 'No es estacionaria'
    else:
        resumen = 'Es estacionaria'
    print(f'[ADF]({results["adf_pvalue"]:.2f})\t\t{name}\t\t{resumen}')
    if results['kpss_pvalue'] > 0.05:
        resumen = 'Es estacionaria'
    else:
        resumen = 'No es estacionaria'
    print(f'[KPSS]({results["kpss_pvalue"]:.2f})\t\t{name}\t\t{resumen}')


def differentiate(series):
    """
    Return differenced series.
    periods: 1 for first difference or seasonal period.
    """
    return series.diff().dropna()

def plot_acf_pacf(series, title=None, lags=30):
    """
    Plot ACF and PACF of the series.
    lags: number of lags to plot.
    """
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    plot_acf(series, lags=lags, ax=ax[0])
    plot_pacf(series, lags=lags, ax=ax[1])
    if title:
        plt.suptitle(title)
    plt.show()


def train_test_split(series, test_size=0.2):
    """
    Split series into train and test sets.
    test_size: fraction for test.
    Returns train, test.
    """
    n = len(series)
    split = int((1 - test_size) * n)
    train = series.iloc[:split]
    test = series.iloc[split:]
    return train, test


def fit_arima(series, order=(1, 0, 1)):
    """
    Fit ARIMA model to series.
    order: (p, d, q)
    Returns fitted model.
    """
    model = ARIMA(series, order=order)
    fitted = model.fit()
    return fitted

def fit_sarima(series, order=(1,1,1), seasonal_order=(1,1,1,7)):
    series = series.asfreq('D')
    model = SARIMAX(series,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    return model.fit(disp=False)


def forecast_arima(fitted, start, steps, periods, freq='D'):
    """
    Forecast using fitted ARIMA model.
    steps: number of periods to forecast.
    Returns a pandas Series of forecasts.
    """
    forecast = fitted.get_forecast(steps=steps, freq=freq)
    forecast_index = pd.date_range(start=start, periods=periods)
    forecast_series = pd.Series(forecast.predicted_mean, index=forecast_index)
    conf_int = forecast.conf_int()
    return forecast_series, conf_int


def plot_arima_forecast(train, test, forecast_series, conf_int, title='ARIMA Forecast', xlabel='Date', ylabel='Demanda (MW)'):
    """
    Plot ARIMA forecasts against actuals.
    train, test: pandas Series
    model: fitted ARIMA
    """
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train, label='Entrenamiento', color='tab:blue')
    plt.plot(test.index, test, label='Prueba', color='tab:orange')
    plt.plot(forecast_series.index, forecast_series, label='Predicciones', color='tab:green')
    plt.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:,  1], color='lightgray', alpha=0.5, label='Intervalo de Confianza')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    plt.show()

def decompose_series(series, model='additive', period=None):
    """
    Decompose series into trend, seasonal, residual.
    period: season length. If None, inferred.
    Returns DecomposeResult.
    """
    return seasonal_decompose(series, model=model, period=period)


def plot_decomposition(decomp, xlabel='Date'):
    """
    Plot decomposition result from seasonal_decompose.
    """
    decomp.plot()
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.show()

def plot_decomposition_2(decomp, title=None, xlabel='Date'):
    trend = decomp.trend.dropna()
    seasonal = decomp.seasonal.dropna()
    residual = decomp.resid.dropna()

    plt.figure(figsize=(12, 10))
    if title:
        plt.suptitle(title)
    plt.subplot(3, 1, 1)
    plt.plot(trend, label='Tendencia', color='tab:blue')
    plt.title('Tendencia')
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(seasonal, label='Estacionalidad', color='tab:orange')
    plt.title('Estacionalidad')
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(residual, label='Residuo', color='tab:green')
    plt.title('Residuo')
    plt.legend()
    plt.tight_layout()
    plt.xlabel(xlabel)
    plt.grid()
    plt.show()

def forecast_prophet(df, periods=12, freq='M'):
    """
    Forecast using Prophet.
    df: DataFrame with columns ['ds', 'y'].
    periods: number of periods to forecast.
    freq: frequency string, e.g. 'D', 'M', 'Y'.
    Returns forecast DataFrame.
    """
    m = Prophet()
    m.fit(df)
    future = m.make_future_dataframe(periods=periods, freq=freq)
    forecast = m.predict(future)
    return forecast


def compare_predictions(actual, predicted):
    """
    Compare actual vs predicted series.
    Returns DataFrame with actual, predicted, and error.
    """
    df = pd.DataFrame({
        'actual': actual,
        'predicted': predicted
    })
    # df['error'] = df['actual'] - df['predicted']
    return df
