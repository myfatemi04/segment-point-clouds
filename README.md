# SAM Point Cloud Segmentation

3 files to segment point clouds using [Segment Anything](https://github.com/facebookresearch/segment-anything).

## Installation

1. Install the required libraries:

```bash
pip install torch transformers opencv-python matplotlib
```

2. Clone the repository:

```bash
git clone https://github.com/myfatemi04/segment-point-clouds.git
cd yourrepository
```

## Usage

1. Ensure your point cloud data is in the correct format (numpy arrays with -10000 for invalid points).
2. Run the `test()` function in the script, which loads sample RGB and point cloud data and demonstrates the segmentation process.

## Example

```python
from sam_point_cloud_segmenter import SamPointCloudSegmenter

# Create an instance of the segmenter
segmenter = SamPointCloudSegmenter(render_2d_results=True)

# Load RGB and point cloud data (not included here, assumed to be in the correct format)
rgbs = [...]  # List of RGB images
pcds = [...]  # List of point clouds (numpy arrays)

# Example segmentation
base_rgb_image = rgbs[0]
base_point_cloud = pcds[0]
supplementary_rgb_images = rgbs[1:]
supplementary_point_clouds = pcds[1:]

bounding_box = [x1, y1, x2, y2]  # Define the bounding box for segmentation
point_cloud, color, segmentation_masks = segmenter.segment(base_rgb_image, base_point_cloud, bounding_box, supplementary_rgb_images, supplementary_point_clouds)

# Further processing and visualization of the segmented point cloud
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

