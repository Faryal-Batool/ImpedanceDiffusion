# Module: Interactive annotation tool for marking start and goal pixels.

import os
import glob
import cv2
import pandas as pd

# ----------------------------
# Config
# ----------------------------
INPUT_DIR = r"D:\Skoltech\Thesis\Diffusion_with_imepdance\diff-gpt\Experiment_annotations"          # folder with your images
OUT_DIR_512 = r"D:\Skoltech\Thesis\Diffusion_with_imepdance\diff-gpt\Experiment_annotations"            # resized images output
EXCEL_PATH = r"D:\Skoltech\Thesis\Diffusion_with_imepdance\diff-gpt\Experiment_annotationsstart_goal_points.xlsx"
POINTS_TXT_PATH = r"D:\Skoltech\Thesis\Diffusion_with_imepdance\diff-gpt\Experiment_annotations\hardcoded_points_1920x1080.txt"

TARGET_W, TARGET_H = 1920, 1080
SMALL_W, SMALL_H = 64, 64

SAVE_RESIZED_IMAGES = True

# Keys
KEY_NEXT = ord('n')    # go to next image (save)
KEY_BACK = ord('b')    # go to previous image
KEY_GOAL = ord('g')    # switch to selecting goal
KEY_RESET = ord('r')   # reset points for current image
KEY_SKIP = ord('s')    # skip image (no save)
KEY_QUIT = ord('q')    # quit

# ----------------------------
# Helpers
# ----------------------------
# Function: List supported image files in annotation order.
def list_images(folder):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.webp")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, e)))
    return sorted(files)

# Function: Clamp a scalar value into the requested range.
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

# Function: Convert full-resolution annotation pixels into 64x64 model-grid pixels.
def to_64(x1920, y1080):
    # scale (1920x1080) -> (64x64)
    x64 = int(round(x1920 * (SMALL_W / TARGET_W)))
    y64 = int(round(y1080 * (SMALL_H / TARGET_H)))
    x64 = clamp(x64, 0, SMALL_W - 1)
    y64 = clamp(y64, 0, SMALL_H - 1)
    return x64, y64

# Function: Draw current start/goal annotations and UI status on an image copy.
def draw_overlay(img, start, goal, mode_text):
    vis = img.copy()

    # Draw start
    if start is not None:
        cv2.circle(vis, start, 6, (0, 255, 0), -1)  # green
        cv2.putText(vis, "START", (start[0] + 8, start[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Draw goal
    if goal is not None:
        cv2.circle(vis, goal, 6, (0, 0, 255), -1)  # red
        cv2.putText(vis, "GOAL", (goal[0] + 8, goal[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # HUD
    hud = [
        f"Mode: {mode_text}",
        "Click = set point",
        "Keys: g=goal mode | r=reset | n=next(save) | b=back | s=skip | q=quit",
    ]
    y = 22
    for line in hud:
        cv2.putText(vis, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 2)
        y += 22

    return vis

# ----------------------------
# Main GUI
# ----------------------------
files = list_images(INPUT_DIR)
if not files:
    raise RuntimeError(f"No images found in: {INPUT_DIR}")

os.makedirs(OUT_DIR_512, exist_ok=True)

# Store results by filename so back/forward editing works
results = {}  # img_name -> dict row

# State
idx = 0
selecting = "start"  # "start" or "goal"
start_pt = None
goal_pt = None

window_name = "Start/Goal Picker (1920x1080)"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, TARGET_W, TARGET_H)  # force window size to match image

# Function: Load the current annotation image and any existing saved points.
def load_image(i):
    global start_pt, goal_pt, selecting
    path = files[i]
    img0 = cv2.imread(path, cv2.IMREAD_COLOR)
    if img0 is None:
        raise RuntimeError(f"Failed to read image: {path}")

    img1920 = cv2.resize(img0, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)

    img_name = os.path.basename(path)

    # Load existing annotations if present
    if img_name in results:
        row = results[img_name]
        start_pt_local = (int(row["1920_x"]), int(row["1920_y"])) if pd.notna(row["1920_x"]) else None
        goal_pt_local = (int(row["1920_goal_x"]), int(row["1920_goal_y"])) if pd.notna(row["1920_goal_x"]) else None
    else:
        start_pt_local, goal_pt_local = None, None

    start_pt = start_pt_local
    goal_pt = goal_pt_local
    selecting = "start" if start_pt is None else ("goal" if goal_pt is None else "start")

    # Debug: confirm size is truly 1920x1080
    # print("Loaded image size (H,W) =", img1920.shape[:2])

    return img_name, img1920

img_name, img1920 = load_image(idx)

# Function: Handle mouse clicks for interactive start/goal annotation.
def mouse_cb(event, x, y, flags, param):
    global start_pt, goal_pt, selecting, img_name, img1920

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if selecting == "start":
        start_pt = (x, y)
        selecting = "goal"
    else:
        goal_pt = (x, y)
        selecting = "start"

        # ✅ AUTO-SAVE immediately once goal is set
        _ = save_row_for_current()
        write_excel()
        write_hardcoded_points_txt()

cv2.setMouseCallback(window_name, mouse_cb)

# Function: Persist the current image annotation into the in-memory table.
def save_row_for_current():
    # Save only if start and goal exist
    if start_pt is None or goal_pt is None:
        return False

    sx, sy = start_pt
    gx, gy = goal_pt
    sx64, sy64 = to_64(sx, sy)
    gx64, gy64 = to_64(gx, gy)

    results[img_name] = {
        "img_name": img_name,

        # 1920x1080 points (actual)
        "1920_x": sx, "1920_y": sy,
        "1920_goal_x": gx, "1920_goal_y": gy,

        # 64x64 points (derived)
        "64_x": sx64, "64_y": sy64,
        "64_goal_x": gx64, "64_goal_y": gy64,
    }
    return True

# Function: Write annotation rows to the configured Excel file.
def write_excel():
    # keep stable order according to files list
    rows = []
    for p in files:
        name = os.path.basename(p)
        if name in results:
            rows.append(results[name])

    df = pd.DataFrame(rows, columns=[
        "img_name",
        "1920_x", "1920_y", "64_x", "64_y",
        "1920_goal_x", "1920_goal_y", "64_goal_x", "64_goal_y"
    ])
    df.to_excel(EXCEL_PATH, index=False)

# Function: Export annotations as Python hardcoded point tuples.
def write_hardcoded_points_txt():
    lines = []
    lines.append("HARDCODED_POINTS = [\n")

    for p in files:
        name = os.path.basename(p)
        if name not in results:
            continue

        row = results[name]
        sx, sy = int(row["1920_x"]), int(row["1920_y"])
        gx, gy = int(row["1920_goal_x"]), int(row["1920_goal_y"])

        comment = os.path.splitext(name)[0]
        lines.append(f"    (({sx}, {sy}), ({gx}, {gy})),  # {comment}\n")

    lines.append("]\n")

    with open(POINTS_TXT_PATH, "w", encoding="utf-8") as f:
        f.writelines(lines)

while True:
    mode_text = "START (click)" if selecting == "start" else "GOAL (press g then click)"
    vis = draw_overlay(img1920, start_pt, goal_pt, mode_text)
    cv2.imshow(window_name, vis)

    key = cv2.waitKey(20) & 0xFF
    if key == 255:
        continue

    if key == KEY_GOAL:
        selecting = "goal"

    elif key == KEY_RESET:
        start_pt = None
        goal_pt = None
        selecting = "start"
        if img_name in results:
            del results[img_name]
        write_excel()
        write_hardcoded_points_txt()

    elif key == KEY_SKIP:
        idx = min(idx + 1, len(files) - 1)
        img_name, img1920 = load_image(idx)

    elif key == KEY_BACK:
        idx = max(idx - 1, 0)
        img_name, img1920 = load_image(idx)

    elif key == KEY_NEXT:
        # save (if complete), export excel, export points txt, optionally save resized image, then next
        _ = save_row_for_current()
        write_excel()
        write_hardcoded_points_txt()

        if SAVE_RESIZED_IMAGES:
            out_path = os.path.join(OUT_DIR_512, img_name)
            cv2.imwrite(out_path, img1920)

        idx = min(idx + 1, len(files) - 1)
        img_name, img1920 = load_image(idx)

    elif key == KEY_QUIT:
        # final save
        save_row_for_current()
        write_excel()
        write_hardcoded_points_txt()
        break

cv2.destroyAllWindows()
print(f"Done. Excel saved to: {EXCEL_PATH}")
print(f"Points TXT saved to: {POINTS_TXT_PATH}")
if SAVE_RESIZED_IMAGES:
    print(f"Resized images saved to: {OUT_DIR_512}")