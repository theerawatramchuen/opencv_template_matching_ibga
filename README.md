# opencv_template_matching_ibga
![image](https://github.com/user-attachments/assets/86f9735d-cd68-42bd-8da1-b5acb539ad1a)

'''python

def my_match_tm_1(image, template, w_crop, h_crop, output_file, search_zone='top_left'):
    import cv2
    import numpy as np

    # Validate crop parameters
    if not (0 <= w_crop <= 1 and 0 <= h_crop <= 1):
        raise ValueError("Crop parameters w_crop and h_crop must be between 0 and 1.")

    try:
        # Load the image and the template as grayscale
        image_full = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        template = cv2.imread(template, cv2.IMREAD_GRAYSCALE)
       
        if image_full is None or template is None:
            raise ValueError("Failed to load image or template.")

        # Crop the middle region of the image
        height, width = image_full.shape
        crop_width = int(width * w_crop)
        crop_height = int(height * h_crop)
        x_start = (width - crop_width) // 2
        y_start = (height - crop_height) // 2
        image_cropped = image_full[y_start:y_start + crop_height, x_start:x_start + crop_width]

        # Define the zones
        half_width = crop_width // 2
        half_height = crop_height // 2

        zones = {
            'top_left': image_cropped[:half_height, :half_width],
            'top_right': image_cropped[:half_height, half_width:],
            'bottom_left': image_cropped[half_height:, :half_width],
            'bottom_right': image_cropped[half_height:, half_width:]
        }

        # Select the zone to search within
        if search_zone not in zones:
            raise ValueError("Invalid search zone. Choose from 'top_left', 'top_right', 'bottom_left', 'bottom_right'.")

        zone_image = zones[search_zone]

        # Perform template matching within the selected zone
        result = cv2.matchTemplate(zone_image, template, cv2.TM_CCOEFF_NORMED)
        
        # Find the location of the best match
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # For TM_SQDIFF and TM_SQDIFF_NORMED, use min_loc
        # For other methods, use max_loc
        top_left = max_loc
        bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])
        
        # Adjust the coordinates to the original cropped image space
        if search_zone == 'top_right':
            top_left = (top_left[0] + half_width, top_left[1])
            bottom_right = (bottom_right[0] + half_width, bottom_right[1])
        elif search_zone == 'bottom_left':
            top_left = (top_left[0], top_left[1] + half_height)
            bottom_right = (bottom_right[0], bottom_right[1] + half_height)
        elif search_zone == 'bottom_right':
            top_left = (top_left[0] + half_width, top_left[1] + half_height)
            bottom_right = (bottom_right[0] + half_width, bottom_right[1] + half_height)

        # Draw a rectangle around the matched region
        cv2.rectangle(image_cropped, top_left, bottom_right, 255, 2)

        # Save the output image
        cv2.imwrite(output_file, image_cropped)

        # Display the result
        cv2.imshow('Matched Image', image_cropped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
          
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

'''
