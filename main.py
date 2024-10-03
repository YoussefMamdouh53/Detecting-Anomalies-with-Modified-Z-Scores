import time
import numpy as np
import matplotlib.pyplot as plt
import statistics
import cv2


def generate_data_stream():
    t = 0
    while True:
        regular = np.random.rand() / 2 + 0.2
        # using sin wave for periodic behavior to the data for seasonal pattern
        seasonal = np.sin(2 * np.pi * t / 100)
        # A small random variation is added to each data point which simulates real-world noise
        noise = np.random.normal(0, 0.2)
        yield regular + seasonal + noise
        t += 1


# Modified Z-Score Function
def calculate_modified_zscore(x, median, mad):
    return 0.6745 * (x - median) / mad


class AnomalyDetector:
    def __init__(self, stream, window_size=100, threshold=0.5):
        self.stream = stream
        self.window_size = window_size
        self.threshold = threshold

    def detect(self):
        data = []

        # Collect initial data for window
        while len(data) < self.window_size:
            data.append(next(self.stream))

        # Compute initial median and MAD
        median = statistics.median(data)
        mad = statistics.median([abs(median - x) for x in data])

        for v in data:
            # median and MAD are calculated for modified Z-score.
            m_zscore = calculate_modified_zscore(v, median, mad)
            # If the absolute Z-score exceeds the threshold, the point is marked as an anomaly.
            yield v, abs(m_zscore) > self.threshold, m_zscore

        while True:
            # Sliding window: For each new data point The old point is removed, and the new point is added to the window
            data.pop(0)
            data.append(next(self.stream))

            # The median and MAD are recalculated, and the modified Z-score for the new point is computed.
            median = statistics.median(data)
            mad = statistics.median([abs(median - x) for x in data])
            m_zscore = calculate_modified_zscore(data[-1], median, mad)

            # If the absolute Z-score exceeds the threshold, the point is marked as an anomaly.
            yield data[-1], abs(m_zscore) > self.threshold, m_zscore


data_stream = generate_data_stream()
detector = AnomalyDetector(data_stream, window_size=1000, threshold=0.4)

points = []
anomalies = []

# Live Visualization

fig = plt.figure(figsize=(16, 8))

while True:
    v, a, zscore = next(detector.detect())

    # Points and their anomaly status are collected continuously
    points.append(v)
    # If the anomaly score exceeds the threshold, the point is colored red, otherwise it's white
    anomalies.append('#FF0000' if a else '#FFFFFF')

    # clearing figure for new draw
    fig.clear()

    # limit visualization window for 50 value by removing the old value
    if len(points) >= 50:
        points.pop(0)
        anomalies.pop(0)

    # Define x, y limits
    plt.xlim(0, 50)
    plt.ylim(-0.5, 1.5)

    # Plotting data
    plt.plot(points)
    plt.scatter(range(len(points)), points, c=anomalies, s=30)

    plt.title("Anomaly Detection Live Preview", fontweight='bold', size='xx-large')
    plt.suptitle('Press ESC to exit', fontweight='bold', size='large')
    plt.legend(['Datastream', 'Anomalies'])
    plt.gca().get_legend().legend_handles[1].set_color('red')

    fig.canvas.draw()

    # converting plot to image
    img_plot = np.array(fig.canvas.renderer.buffer_rgba())
    # convert color mode from rgb to bgr for opencv
    img_plot = cv2.cvtColor(img_plot, cv2.COLOR_RGBA2BGR)

    # Displaying the Image:
    cv2.imshow("Anomaly Detection Live Preview", img_plot)

    # adding delay between each frame
    time.sleep(0.2)

    # close window if user pressed ESC key
    if cv2.waitKey(1) == 27:
        break
