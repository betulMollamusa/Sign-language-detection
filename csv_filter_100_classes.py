import pandas as pd

df = pd.read_csv("val_labels.csv", header=None, names=["video_id", "label"])
first_100_labels = sorted(df["label"].unique())[:100]
filtered_df = df[df["label"].isin(first_100_labels)]
filtered_df.to_csv("val_labels_first_100.csv", index=False, header=False)
