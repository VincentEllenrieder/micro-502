import cv2
import numpy as np
import matplotlib.pyplot as plt

detector = 'polygon'

def detect_gate_corners(image_path):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found or path is incorrect")

    # Preprocess
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Find contours
    contours, _ = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)

    if detector == 'polygon':
        # Approximate contour to polygon
        perim = cv2.arcLength(contour, True)
        for factor in np.linspace(0.001, 0.1, 100):
            eps = factor * perim
            approx = cv2.approxPolyDP(contour, eps, True).reshape(-1, 2)
            if len(approx) == 4:
                corners = approx
                break

        # If the contour has 4 corners (quadrilateral)
        print("There are {} corners".format(len(approx)))
        # Draw the full contour in green
        cv2.drawContours(img, [contour], -1, (0, 255, 0), 1)

        # Draw the corners in blue
        for p, point in enumerate(corners):
            #color_coeff = np.mod(p, len(approx))
            corner = tuple(point)
            img[corner[1], corner[0]] = (255, 0, 0)  # Set pixel to blue (BGR)
            

        # Show result
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Gate corners and contours")
        plt.axis('off')
        plt.show()

    if detector == 'rect_fit':
        rect = cv2.minAreaRect(contour)           # ((cx, cy), (w, h), angle)
        box = cv2.boxPoints(rect)             # 4×2 float32 array
        box = np.int0(box)                    # convert to int

        # 4. Draw everything
        vis = img.copy()
        # 4a. original contour in green
        cv2.drawContours(vis, contour, -1, (0,255,0), 1)
        # 4b. fitted rectangle in blue
        cv2.drawContours(vis, [box], -1, (255,0,0), 1)
        # 4c. mark corners with small red squares
        for (x,y) in box:
            x, y = int(x), int(y)
            cv2.rectangle(vis,
                        (x-3, y-3),
                        (x+3, y+3),
                        (0,0,255),  # red BGR
                        thickness=-1)

        # 5. Display
        plt.figure(figsize=(6,6))
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.title("Contour (green), Rectangle (blue), Corners (red)")
        plt.axis('off')
        plt.show()

    if detector == 'convex_hull':
        hull = cv2.convexHull(contour).reshape(-1,2)   # shape (M,2)

        # compute angles of hull pts around centroid
        c = hull.mean(axis=0)
        angs = np.arctan2(hull[:,1]-c[1], hull[:,0]-c[0])
        angs = (angs + 2*np.pi) % (2*np.pi)

        # sort hull points by angle
        order = np.argsort(angs)
        hull_s = hull[order]
        angs_s = angs[order]

        # compute angular gaps (circular)
        gaps = np.diff(np.concatenate([angs_s, angs_s[:1]+2*np.pi]))
        # find the four largest gaps
        idxs = np.argpartition(gaps, -4)[-4:]
        # the corner occurs at the start of each gap: next point in sorted list
        corners = []
        M = len(hull_s)
        for i in idxs:
            corner = hull_s[(i+1) % M]
            corners.append(corner)
        corners = np.array(corners)  # shape (4,2)

        # draw for visualization
        vis = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(vis, [hull_s.reshape(-1,1,2)], -1, (0,255,0), 2)      # hull
        cv2.drawContours(vis, [corners.reshape(-1,1,2)], -1, (255,0,0), 2)    # quad
        for x,y in corners:
            xi, yi = int(x), int(y)
            if 0 <= yi < vis.shape[0] and 0 <= xi < vis.shape[1]:
                vis[yi, xi] = (0, 0, 255)  # single red pixel

        plt.figure(figsize=(6,6))
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.title("Hull (green), Quad (blue), Corners (red)")
        plt.axis('off')
        plt.show()



    

if __name__ == "__main__":
    detector = 'polygon'
    detect_gate_corners("C:/Documents_LOCAL/EPFL/MA2/Aerial_robotics/micro-502/controllers/main/image_analysis/thresholding_image.png")