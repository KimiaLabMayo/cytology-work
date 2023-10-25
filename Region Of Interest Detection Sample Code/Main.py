# By Rohollah Moosavi Tayebi, email: rohollah.moosavi@uwaterloo.ca/moosavi.tayebi@gmail.com

# This script gets a raw aspirate WSI and detect Region of Interest (ROI)
# tiles inside that

from Tile_Selection import tile_selection_module

weight_path = "Region of Interest Detection Weights"
wsi_path = "path to WSI"
ROI_path = "output path"
Tile_Text_File = "Tiles_path.txt"
grid_size = [80, 120] # width, height
tile_size = 512
down_scale = 1
threshold = 0.85
model_name = 'densenet121'

def tile_selection_method():
    tile_selection_module(ROI_path, Tile_Text_File, tile_size, down_scale, threshold, model_name, grid_size, wsi_path, weight_path)

tile_selection_method()