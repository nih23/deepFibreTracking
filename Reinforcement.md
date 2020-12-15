# Reinforcement Learning environment
There are a few classes and methods which could help you building your reinforcement learning environment.

## DWI Data representation
Firstly, you can use `DataContainer` objects to store and retrieve DWI data:

### Initialization

```python
from src.data import HCPDataContainer, ISMRMDataContainer
hcp_data = HCPDataContainer(100307)
ismrm_data = ISMRMDataContainer()
```
### Normalizing and Cropping

```python
ismrm_data.crop().normalize()
hcp_data.crop().normalize()
```

### Retrieving the FA:
```python
fa = hcp_data.get_fa() # I am not sure if it is better to calculate FA before cropping/normalizing or after
```
Keep in mind that any tracking should be done before applying the crop and normalize operations because they alter the MRI values.

### Coordinate system transforms:
```python
ijk_points = hcp_data.to_ijk(ras_points) # Transform to Image coordinate system
ras_points = hcp_data.to_ras(ijk_points) # Transform to World RAS+ coordinate system

ras_points == hcp_data.to_ras(hcp_data.to_ijk(ras_points)) # True
```

### DWI interpolation
You can retrieve interpolated DWI values with the following method. If you pass `ignore_outside_points=True`, there won't be an error thrown for points outside of the DWI Image.
```python
interpolated_dwi = hcp_data.get_interpolated_dwi(ras_points, ignore_outside_points=False)
```

### Fields
The fields can be helpful for checks or additional calculations based on the loaded data.
```python
hcp_data.options # configuration 
hcp_data.path # file path to loaded data
hcp_data.data # the actual MRI Data as SimpleNamespace
```

## Tracking and Streamline Representation

You use `Tracker` objects to represent already tracked streamlines or to track streamlines using CSD or DTI:

### Loading Ground Truth Streamlines
You can retrieve tracked `Tracker` Objects in multiple ways:
```python
ismrm_sl = ISMRMReferenceStreamlinesTracker(ismrm_data, streamline_count=10000)
ismrm_sl.track()

file_sl = StreamlinesFromFileTracker("streamlines.vtk")
ismrm_sl.track()

hcp_sl = CSDTracker(hcp_data, random_seeds=True, fa_threshold=0.15)
ismrm_sl.track()
```
Please keep in mind that the `CSDTracker` and `DTITracker` have internal caches, so your given DWI containers aren't tracked each time when you call the track method, but only if there is no corresponding cache file or the cache is deactivated. Because the cache operates on names and paths, it is important that you don't replace DWI files with others with identical names and paths without deleting the cache.

### Helpful methods
```python
streamlines = file_sl.get_streamlines() # retrieve the actual streamlines
filtered_streamlines = file_sl.filtered_streamlines_by_length(minimum=70) # filter streamlines

hcp_sl.save_to_file("hcp_streamlines.vtk")
```

[TODO add Tracking and retrieving streamlines example]::

## Config

Furthermore, you can use the `Config` if you want to read and write your own parameters:

Get the singletone with

```python
config = Config.get_config()
```

You can read and set attributes:
```python
config.set("section", "option1", value="value")
        
string_config = config.get("section1", "option2", fallback="default")
int_config = config.getint("section1", "numerical_value", fallback="0")
float_config = config.getfloat("section", "option_f", fallback="1.2")
bool_config = config.getboolean("section", "option_b", fallback="True")
```
Loading and saving is handled automatically, and the fallback values are being added to the configuration file as soon as they are requested the first time.

## Helpful methods to use for you
Last, I gathered a few methods which should assist you in creating your training environment without having to reinvent the wheel regarding the data processing:

### 1. data_container.get_interpolated_dwi(points, ignore_outside_points=False)
Returns the interpolated DWI at the given points while keeping the dimensions of the given points, for example, you can put in a point ndarray of size `A x B X 3` and you get an ndarray of the size `A x B x DWI`  
### 2. util.get_grid(grid_dimension)
Takes a 3D tuple `(x,y,z)` as `grid_dimension` and generates a grid with the given dimensions, applyable to any point or:
### 3. util.apply_rotation_matrix_to_grid(grid, rot_matrix)
Takes a grid from the `util.get_grid` method and a list of rotation_matrices and applies all the rotations to the grid parallelized, returning a list of grids. 
### 4. util.direction_to_classification(sphere, next_dir, include_stop=False, last_is_stop=False, stop_values=None)
Takes a `dipy` Sphere with directions, and a list of directions and will return the classifier output weighted after the similarity to the given vector. If `include_stop` is `True`, you can either provide `last_is_stop` which defines that the last element fulfills the stop condition or provide a stop_value between 0 and 1 for every next_dir in `stop_values`, which will be added to the classifier output.
### 5. processing.calculate_item(data_container, previous_sl, next_dir)
Takes a `DataContainer`, the streamline calculated until this point and the direction it is supposed to interpolate to. Returns an `(input, output)` tuple for the NN. Available for every processing method.