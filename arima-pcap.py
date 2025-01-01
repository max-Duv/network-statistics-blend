# Libraries used
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error


# Reads PCAP file and extracts TCP traffic data
def read_pcap(file_path):
    import pyshark
    cap = pyshark.FileCapture(file_path, display_filter='tcp')
    timestamps = []
    for packet in cap:
        timestamps.append(float(packet.sniff_timestamp))
    cap.close()
    return pd.Series(timestamps)

# Create ARIMA model and detect anomalies
def detect_anomalies(tcp_data):
    tcp_data = tcp_data.resample('s').count()

    # Fit ARIMA model
    model = ARIMA(tcp_data, order=(5, 1, 0))
    model_fit = model.fit()
    predictions = model_fit.predict(start=0, end=len(tcp_data) - 1, dynamic=False)
    mse = mean_squared_error(tcp_data, predictions)

    # Detect anomalies
    anomalies = np.abs(tcp_data - predictions) > 2 * np.sqrt(mse)
    return tcp_data, predictions, anomalies


# visualization
def visualize_results(tcp_data, predictions, anomalies):
    plt.figure(figsize=(12, 6))
    plt.plot(tcp_data.index, tcp_data, label='TCP Traffic')
    plt.plot(tcp_data.index, predictions, color='red', label='Predicted Traffic')
    plt.scatter(tcp_data.index[anomalies], tcp_data[anomalies], color='orange', label='Anomalies')
    plt.xlabel('Time (since start)')
    plt.ylabel('Number of Packets')
    plt.title('TCP Traffic and Anomalies Detection')
    plt.legend()
    plt.show()

# functionality

file_path = 'ARIMA.DEC.NYE.NIGHT.pcap'
tcp_data = read_pcap(file_path)
tcp_data.index = pd.to_datetime(tcp_data, unit='s')
tcp_data, predictions, anomalies = detect_anomalies(tcp_data)
visualize_results(tcp_data, predictions, anomalies)