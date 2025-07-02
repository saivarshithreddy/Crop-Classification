
""" NAME : B.SAI VARSHITH REDDY """
"""Approach
I used two different approaches.
The first approach involved training with 3 set of features:

1.Image pixel values
2.About 10 vegetation/spectral indices (e.g. NDVI, AVI etc.), and their relevant statistics
3.Spatial features (e.g area of farm etc.).
The second approach involved training with only pixel values, and their relevant statistics.
My solution is an ensemble weighted average of the two approaches.

Modelling
The two approaches each went through the same modelling process by using a CatboostClassifier (without class_weights), another CatboostClassifier (with class_weights to take care of class imbalance), and a LinearDiscriminant algorithm (known in sklearn as LinearDiscriminantAnalysis - LDA ). LDA is a weak learner, so in order to improve it's performance, I bagged (ensemble) it using sklearn's BaggingClassifier. The weighted Catboost and bagged LDA added some diversity to the modelling due to the highly imbalanced dataset, and in general improved performance.

"""
""" REQUIREMENTS : sklearn==0.22.2
                  eli5==0.10.1
                  catboost==0.22
                  scipy==1.4.1
                  numpy==1.18.3
                  pandas==1.0.3    """

# Required libraries
import requests
from urllib.parse import urlparse
from pathlib import Path
from datetime import datetime

API = 'YOUR KEY HERE'

!mkdir data

output_path = Path("data/")

# these headers will be used in each request
headers = {
    'Authorization': f'Bearer {API}',
    'Accept':'application/json'
}

def get_download_url(item, asset_key, headers):
    asset = item.get('assets', {}).get(asset_key, None)
    if asset is None:
        print(f'Asset "{asset_key}" does not exist in this item')
        return None
    r = requests.get(asset.get('href'), headers=headers, allow_redirects=False)
    return r.headers.get('Location')

def download_label(url, output_path, tileid):
    filename = urlparse(url).path.split('/')[-1]
    outpath = output_path/tileid
    outpath.mkdir(parents=True, exist_ok=True)
    
    r = requests.get(url)
    f = open(outpath/filename, 'wb')
    for chunk in r.iter_content(chunk_size=512 * 1024): 
        if chunk:
            f.write(chunk)
    f.close()
    print(f'Downloaded {filename}')
    return 

def download_imagery(url, output_path, tileid, date):
    filename = urlparse(url).path.split('/')[-1]
    outpath = output_path/tileid/date
    outpath.mkdir(parents=True, exist_ok=True)
    
    r = requests.get(url)
    f = open(outpath/filename, 'wb')
    for chunk in r.iter_content(chunk_size=512 * 1024): 
        if chunk:
            f.write(chunk)
    f.close()
    print(f'Downloaded {filename}')
    return

# paste the id of the labels collection:
collectionId = 'ref_african_crops_kenya_02_labels'

# these optional parameters can be used to control what items are returned. 
# Here, we want to download all the items so:
limit = 100 
bounding_box = []
date_time = []

# retrieves the items and their metadata in the collection
r = requests.get(f'https://api.radiant.earth/mlhub/v1/collections/{collectionId}/items', params={'limit':limit, 'bbox':bounding_box,'datetime':date_time}, headers=headers)
collection = r.json()

# retrieve list of features (in this case tiles) in the collection
for feature in collection.get('features', []):
    assets = feature.get('assets').keys()
    print("Feature", feature.get('id'), 'with the following assets', list(assets))

for feature in collection.get('features', []):
    
    tileid = feature.get('id').split('tile_')[-1][:2]

    # download labels
    download_url = get_download_url(feature, 'labels', headers)
    download_label(download_url, output_path, tileid)
    
    #download field_ids
    download_url = get_download_url(feature, 'field_ids', headers)
    download_label(download_url, output_path, tileid)

# paste the id of the imagery collection:
collectionId = 'ref_african_crops_kenya_02_source'

# these optional parameters can be used to control what items are returned. 
# Here, we want to download all the items so:
limit = 100 
bounding_box = []
date_time = []

# retrieves the items and their metadata in the collection
r = requests.get(f'https://api.radiant.earth/mlhub/v1/collections/{collectionId}/items', params={'limit':limit, 'bbox':bounding_box,'datetime':date_time}, headers=headers)
collection = r.json()

# List assets of the items
for feature in collection.get('features', []):
    assets = feature.get('assets').keys()
    print(list(assets))
    break #all the features have the same type of assets. for simplicity we break the loop here.

# This cell downloads all the multi-spectral images throughout the growing season for this competition.
# The size of data is about 1.5 GB, and download time depends on your internet connection. 
# Note that you only need to run this cell and download the data once.
i = 0
for feature in collection.get('features', []):
    assets = feature.get('assets').keys()
    tileid = feature.get('id').split('tile_')[-1][:2]
    date = datetime.strftime(datetime.strptime(feature.get('properties')['datetime'], "%Y-%m-%dT%H:%M:%SZ"), "%Y%m%d")
    for asset in assets:
        i += 1
        if i > 0: # if resuming after it failed
          download_url = get_download_url(feature, asset, headers)
          download_imagery(download_url, output_path, tileid, date)

"""# Opening files

Again, this is mainly code from the starter notebook shared to open the tiff files as arrays. I add in a way to view the sentinel images as RGB since that's pretty cool, but it isn't actually useful for our purposes.
"""

!pip install tifffile

# Commented out IPython magic to ensure Python compatibility.
# Required libraries
import tifffile as tiff
import datetime
import matplotlib.pyplot as plt
# %matplotlib inline

def load_file(fp):
    """Takes a PosixPath object or string filepath
    and returns np array"""
    
    return tiff.imread(fp.__str__())

# List of dates that an observation from Sentinel-2 is provided in the training dataset
dates = [datetime.datetime(2019, 6, 6, 8, 10, 7),
         datetime.datetime(2019, 7, 1, 8, 10, 4),
         datetime.datetime(2019, 7, 6, 8, 10, 8),
         datetime.datetime(2019, 7, 11, 8, 10, 4),
         datetime.datetime(2019, 7, 21, 8, 10, 4),
         datetime.datetime(2019, 8, 5, 8, 10, 7),
         datetime.datetime(2019, 8, 15, 8, 10, 6),
         datetime.datetime(2019, 8, 25, 8, 10, 4),
         datetime.datetime(2019, 9, 9, 8, 9, 58),
         datetime.datetime(2019, 9, 19, 8, 9, 59),
         datetime.datetime(2019, 9, 24, 8, 9, 59),
         datetime.datetime(2019, 10, 4, 8, 10),
         datetime.datetime(2019, 11, 3, 8, 10)]

bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12', 'CLD']

# Sample file to load:
file_name = "data/00/20190825/0_B03_20190825.tif"
band_data = load_file(file_name)

fig = plt.figure(figsize=(7, 7))
plt.imshow(band_data, vmin=0, vmax=0.15)

# Quick way to see an RGB image. Can mess with the scaling factor to change brightness (3 in this example)

import numpy as np
def load_rgb(tile, date):

  r = load_file(f"data/{tile}/{date}/{tile[1]}_B04_{date}.tif")
  g = load_file(f"data/{tile}/{date}/{tile[1]}_B03_{date}.tif")
  b = load_file(f"data/{tile}/{date}/{tile[1]}_B02_{date}.tif")
  arr = np.dstack((r, g, b))
  print(max(g.flatten()))
  return arr

fig, ax = plt.subplots(figsize=(12, 18))
ax.imshow(load_rgb('01', '20190825')*3)
plt.tight_layout()

"""# Getting data for each pixel in fields

This is where we start massaging the data into a format we can use. I read in the labels, find the pixel locations of all the fields (most are only a few pixels in size) and store the pixel locations, the field ID and the label.

Then we loop through all the images, sampling the different image bands to get the values for each pixel in each field. This isn't a very clean way to do this, but it works and doesn't take long to run in this case :)
"""

# Not super efficient but  ¯\_(ツ)_/¯
import pandas as pd

row_locs = []
col_locs = []
field_ids = []
labels = []
tiles = []
for tile in range(4):
  fids = f'/content/data/0{tile}/{tile}_field_id.tif'
  labs = f'/content/data/0{tile}/{tile}_label.tif'
  fid_arr = load_file(fids)
  lab_arr = load_file(labs)
  for row in range(len(fid_arr)):
    for col in range(len(fid_arr[0])):
      if fid_arr[row][col] != 0:
        row_locs.append(row)
        col_locs.append(col)
        field_ids.append(fid_arr[row][col])
        labels.append(lab_arr[row][col])
        tiles.append(tile)

df = pd.DataFrame({
    'fid':field_ids,
    'label':labels,
    'row_loc': row_locs,
    'col_loc':col_locs,
    'tile':tiles
})

print(df.shape)
print(df.groupby('fid').count().shape)
df.head()

# Sample the bands at different dates as columns in our new dataframe
col_names = []
col_values = []

for tile in range(4): # 1) For each tile
  print('Tile: ', tile)
  for d in dates: # 2) For each date
    print(str(d))
    d = ''.join(str(d.date()).split('-')) # Nice date string
    t = '0'+str(tile)
    for b in bands: # 3) For each band
      col_name = d+'_'+b
      
      if tile == 0:
        # If the column doesn't exist, create it and populate with 0s
        df[col_name] = 0

      # Load im
      im = load_file(f"data/{t}/{d}/{t[1]}_{b}_{d}.tif")
      
      # Going four levels deep. Each second on the outside is four weeks in this loop
      # If we die here, there's no waking up.....
      vals = []
      for row, col in df.loc[df.tile == tile][['row_loc', 'col_loc']].values: # 4) For each location of a pixel in a field
        vals.append(im[row][col])
      df.loc[df.tile == tile, col_name] = vals
df.head()

df.to_csv('sampled_data.csv', index=False)

!cp 'sampled_data.csv' 'drive/My Drive/' # You can mount drive and save like this, or just download sampled_data.csv and use that later.

"""# Trying a few simple models

Now that we have our data in a nice tabular data format, we can try some simple models. I'll restart the runtime here and load in the data we saved earlier. If you have already done the first two steps, you can just upload sampled_data.csv and start from here.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss

# Load the data
df = pd.read_csv('sampled_data.csv')
print(df.shape)
df.head()

# Split into train and test sets
train = df.loc[df.label != 0]
test =  df.loc[df.label == 0]
train.shape, test.shape

# Split train into train and val, so that we can score our models locally (test is the test set)

# Splitting on field ID since we want to predict for unseen FIELDS
val_field_ids = train.groupby('fid').mean().reset_index()['fid'].sample(frac=0.3).values
tr = train.loc[~train.fid.isin(val_field_ids)].copy()
val = train.loc[train.fid.isin(val_field_ids)].copy()

# Split into X and Y for modelling
X_train, y_train = tr[tr.columns[5:]], tr['label']
X_val, y_val = val[val.columns[5:]], val['label']

"""### We can predict on pixels and then combine, or group into fields and then predict. Let's try both and compare!"""

# Predicting on pixels

# Fit model (takes a few minutes since we have plenty of rows)
model = RandomForestClassifier(n_estimators=500)
model.fit(X_train.fillna(0), y_train)

# Get predicted probabilities
preds = model.predict_proba(X_val.fillna(0))

# Add to val dataframe as columns
for i in range(7):
  val[str(i+1)] = preds[:,i]

# Get 'true' vals as columns in val
for c in range(1, 8):
  val['true'+str(c)] = (val['label'] == c).astype(int)

pred_cols = [str(i) for i in range(1, 8)]
true_cols = ['true'+str(i) for i in range(1, 8)]

# Score
print('Pixel score:', log_loss(val[true_cols], val[pred_cols])) # Note this isn't the score you'd get for a submission!!
print('Field score:', log_loss(val.groupby('fid').mean()[true_cols], val.groupby('fid').mean()[pred_cols])) # This is what we'll compare to

# Inspect visually
val[['label']+true_cols+pred_cols].head()

# Predicting on fields (grouping first)

# Group
train_grouped = tr.groupby('fid').mean().reset_index()
val_grouped = val.groupby('fid').mean().reset_index()
X_train, y_train = train_grouped[train_grouped.columns[5:]], train_grouped['label']
X_val, y_val = val_grouped[train_grouped.columns[5:]], val_grouped['label']

# Fit model
model = RandomForestClassifier(n_estimators=500)
model.fit(X_train.fillna(0), y_train)

# Get predicted probabilities
preds = model.predict_proba(X_val.fillna(0))

# Add to val_grouped dataframe as columns
for i in range(7):
  val_grouped[str(i+1)] = preds[:,i]

# Get 'true' vals as columns in val
for c in range(1, 8):
  val_grouped['true'+str(c)] = (val_grouped['label'] == c).astype(int)

pred_cols = [str(i) for i in range(1, 8)]
true_cols = ['true'+str(i) for i in range(1, 8)]
val_grouped[['label']+true_cols+pred_cols].head()

# Already grouped, but just to double check:
print('Field score:', log_loss(val_grouped.groupby('fid').mean()[true_cols], val_grouped.groupby('fid').mean()[pred_cols]))

"""### Predicting on fields is much faster, and a little more accurate. But feel free to experiment :)

# Making a Submission

Let's take our favourite model, make predictions and submit on Zindi!
"""

# Group test as we did for val
test_grouped = test.groupby('fid').mean().reset_index()
preds = model.predict_proba(test_grouped[train_grouped.columns[5:]])

prob_df = pd.DataFrame({
    'Field_ID':test_grouped['fid'].values
})
for c in range(1, 8):
    prob_df['Crop_ID_'+str(c)] = preds[:,c-1]
prob_df.head()

# Check the sample submission and compare
ss = pd.read_csv('SampleSub.csv')
ss.head()

# Merge the two, to get all the required field IDs
ss = pd.merge(ss['Field_ID'], prob_df, how='left', on='Field_ID')
print(ss.isna().sum()['Crop_ID_1']) # Missing fields
# Fill in a low but non-zero val for the missing rows:
ss = ss.fillna(1/7) # There are 34 missing fields
ss.to_csv('starter_nb_submission.csv', index=False)





