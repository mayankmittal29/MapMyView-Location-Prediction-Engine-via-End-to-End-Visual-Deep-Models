import pandas as pd

# Load training CSV
df = pd.read_csv("labels_train_updated.csv")  # Replace with the path to your actual CSV

# Compute mean and std for latitude and longitude
LAT_MEAN = df["latitude"].mean()
LAT_STD = df["latitude"].std()
LON_MEAN = df["longitude"].mean()
LON_STD = df["longitude"].std()
