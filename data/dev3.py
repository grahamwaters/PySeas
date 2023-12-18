def split_into_panels(image, number_of_panels=6):
    # Split the image into six equal vertical panels (side by side)
    width = image.shape[1]
    panel_width = width // number_of_panels
    panels = [image[:, i*panel_width:(i+1)*panel_width] for i in range(number_of_panels)]
    # Ensure last panel takes any remaining pixels to account for rounding
    panels[-1] = image[:, (number_of_panels-1)*panel_width:]
    return panels

def stitch_panels_horizontally(panels):
    # Ensure all panels are the same height before stitching
    max_height = max(panel.shape[0] for panel in panels)
    panels_resized = [cv2.resize(panel, (panel.shape[1], max_height), interpolation=cv2.INTER_LINEAR) for panel in panels]
    return np.concatenate(panels_resized, axis=1)

def stitch_vertical(rows):
    # Ensure all rows are the same width before stitching
    max_width = max(row.shape[1] for row in rows)
    # Resize rows to the max width or pad with black pixels
    rows_resized = []
    for row in rows:
        if row.shape[1] < max_width:
            padding = np.zeros((row.shape[0], max_width - row.shape[1], 3), dtype=np.uint8)
            row_resized = np.concatenate((row, padding), axis=1)
        else:
            row_resized = row
        rows_resized.append(row_resized)

    # Stitch the rows together
    return np.concatenate(rows_resized, axis=0)
