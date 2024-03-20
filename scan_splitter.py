import cv2, re, argparse, pytesseract
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--infile", help="Input file path", required=True)
parser.add_argument("--outdir", help="Output directory path", required=True)
parser.add_argument("--text-detect", help="Chose if you want to detect text header (OCR)", choices=["yes","no"], default="no")
parser.add_argument("--write-header", help="Do you want to write the header image ?", choices=["yes","no"], default="no")
parser.add_argument("--write-leafs", help="Do you want to write the image of 5 leaf without header ?", choices=["yes","no"], default="no")

args = parser.parse_args()

outputPath= "187.png"
infile = args.infile
outdir = args.outdir

print(f"Processing {infile} ...")

text_detect = args.text_detect
write_header = args.write_header
write_leafs = args.write_leafs


infile_name = infile.split("/")[-1].split(("."))[0]
infile_name = infile_name.replace('(','').replace(')','')


# ================================
#   Prepare image
# ================================


img = cv2.imread(infile)
h, w = img.shape[:2]


# median blur
median = cv2.medianBlur(img, 5)

# Seuillage
lower = (0,0,0)
upper = (15,15,15)
thresh = cv2.inRange(median, lower, upper)

# Find square object
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# Merge separated square object 
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (29,29))
morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)

# Calculate the histogram of each column
hist_col = np.zeros((img.shape[1], 256))
hist_row = np.zeros((img.shape[0], 256))


# ================================
#   UPDATE FILENAME 
# ================================
if text_detect == "yes":
    custom_config = r'--oem 3 --psm 12'
    string = pytesseract.image_to_string(img[:int(h/3),int(w/2):], config=custom_config)
    string_list = [re.sub('[^a-zA-Z0-9]+', '_', _) for _ in list(string)]
    string_list = "".join(string_list).split(("_"))
    s_idx = string_list.index("EPO")
    infile_name = "".join(string_list[s_idx]+"_"+string_list[s_idx+1])


# ================================
#   Find horizontal line
# ================================

# iter rows
for i in range(morph.shape[0]):
    hist_row[i] = cv2.calcHist([morph[i,:]], [0], None, [256], [0,256]).flatten()

# Get the minimum black value for rows
r_black = hist_row[:,0]
r_min_idx = list(r_black).index(min(r_black))

# Split image top and down
# 5mm = 1200px/5 = 240px
img_up = img[:r_min_idx-240,:]
if write_header == "yes":
    cv2.imwrite(f"{outdir}/{infile_name}_HEADER.jpg",img_up)

img_down = img[r_min_idx+240:,:]
morph_down = morph[r_min_idx+240:,:]
if write_header == "yes":
    cv2.imwrite(f"{outdir}/{infile_name}_BODY.jpg",img_down)


# ================================
#   Find the vertical black lines
# ================================

verti_lines_lim = {
    1:{
        "s":0,
        "e":800
    },
    2:{
        "s":1700,
        "e":2500
    },
    3:{
        "s":3700,
        "e":4500
    },
    4:{
        "s":5600,
        "e":6400
    },
    5:{
        "s":7600,
        "e":8500
    },
    6:{
        "s":9500,
        "e":10336
    }
}

# iter columns
for i in range(morph_down.shape[1]):
    hist_col[i] = cv2.calcHist([morph_down[:,i]], [0], None, [256], [0,256]).flatten()

verti_lines = list()
for pos in verti_lines_lim:
    s = verti_lines_lim[pos]["s"]
    e = verti_lines_lim[pos]["e"]
    
    c_black = hist_col[s:e,-1]
    c_min_idx = list(c_black).index(max(c_black)) 
    
    c_black_idx=[]
    for i, val in enumerate(c_black):
        if val > 0:
            c_black_idx.append(i) 
    if len(c_black_idx) < 1: 
        if pos == 1: verti_lines.append((0,60))
        else: verti_lines.append((10276,10336))
    else:
        verti_lines.append((min(c_black_idx)+s,max(c_black_idx)+s))

# ================================
#   Function to extract leaf ONLY
# ================================


def extract_leaf(img):
    # Define lower and uppper limits
    lower = np.array([200, 200, 200])
    upper = np.array([255, 255, 255])
    # Create mask to only select black
    thresh = cv2.inRange(img, lower, upper)
    # apply morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # invert colors
    morph = 255-morph
    # Find contours in the binary image
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Sort the contours by area in descending order and find the largest
    largest_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    # Get the coordinates
    x, y, w, h = cv2.boundingRect(largest_contour)
    return img[y:y+h,x:x+w]


# ================================
#   Cut vertical images
# ================================

images_lim = dict()
for nb in range(1,6):
    images_lim[nb] = {
        "x1": verti_lines[nb-1][1],
        "x2": verti_lines[nb][0]
    }

for i in images_lim:
    x1 = images_lim[i]["x1"]
    x2 = images_lim[i]["x2"]
    leaf = extract_leaf(img_down[:,x1:x2])
    cv2.imwrite(f"{outdir}/{infile_name}_leaf_{i}.jpg",leaf)
